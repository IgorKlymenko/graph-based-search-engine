"""Deterministic span inventory for an episode, and the merge back from the LLM.

The pipeline never lets a model hand back rewritten text or raw offsets. Instead:

1. ``parse_spans`` finds what a parser can find and issues ``c*`` ids.
2. ``mark_text`` renders a *copy* of the episode with those ids inlined. The
   clean text is untouched, so every offset stays true to the stored episode.
3. The model works on the marked copy and answers in ids. Spans the parser
   missed come back as ``n*`` with verbatim text, which ``locate_new_spans``
   re-finds in the clean text — a span that does not match exactly once is
   dropped and counted rather than guessed at.
4. ``merge_spans`` collapses overlaps (longest wins) into one lookup table.

Everything here is deterministic and offline; the only LLM contact is the schema
in ``prompts.py`` and the orchestration in ``1_get_candidates.py``.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import NamedTuple

_NLP = None

# Candidate types. ``pronoun`` and ``definite_description`` drive coreference
# downstream, so they are derived from the parse rather than asked of the model:
# a field that is load-bearing should not be able to come back missing.
PROPER_NAME = "proper_name"
DEFINITE_DESCRIPTION = "definite_description"
PRONOUN = "pronoun"
NOMINAL = "nominal"

_DEFINITE = {"the", "this", "that", "these", "those"}
_HAS_LETTER = re.compile(r"[^\W\d_]")
_OPEN, _CLOSE = "⟦", "⟧"

# Words that cannot name anything on their own: pronouns, bare determiners,
# quantifiers, and number words. The list is closed on purpose — the derived
# ``candidate_type`` is not reliable enough to filter on, having typed "syncs"
# and "both" as pronouns in real runs.
_NON_IDENTIFYING = frozenset(["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "one", "oneself", "this", "that", "these", "those", "which", "who", "whom", "whose", "what", "whatever", "whoever", "whichever", "the", "a", "an", "both", "all", "some", "any", "each", "every", "either", "neither", "few", "many", "much", "several", "most", "none", "no", "other", "others", "another", "same", "such", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "first", "second", "third", "fourth", "fifth", "last", "next", "previous"])


def is_identifying(text: str) -> bool:
    """Can this surface name something, or is it purely grammatical scaffolding?

    False only when *every* token is a function word or a bare numeral, so the
    filter removes "the", "three", "it", "both" and keeps anything carrying a
    content word. That line is deliberate rather than aesthetic: "frozen" or
    "the freeze" may look thin, but in a corpus about a database they are real
    query keys with implied meaning, and a surface a user might type is worth
    indexing however unglamorous it looks. "the" is not a query key in any
    corpus.

    Kept: "frozen", "the freeze", "database", "GPT-4", "Q3", "the pilot tenant".
    Dropped: "the", "three", "it", "they", "both", "2023", "this".
    """
    tokens = re.findall(r"[^\W_]+", text.lower())
    if not tokens:
        return False
    return not all(token in _NON_IDENTIFYING or token.isdigit()
                   for token in tokens)


# Markdown scaffolding that carries no meaning for a parser. Line-leading only,
# so a hyphen inside "auto-enrollment" survives; pipes are blanked anywhere,
# since a table cell boundary is never part of a name.
_LINE_MARKER = re.compile(r"^[ \t]*(?:[#>*+\-]+|\d+[.)])", re.MULTILINE)
_ALWAYS_BLANK = "|`"

# Notes punctuation that separates clauses without being sentence punctuation.
# The parser reads "-->" as an ordinary token, so a noun chunk runs straight
# through it and swallows the next clause; treating these as breaks keeps the
# chunk boundaries where a reader would put them.
_CLAUSE_BREAK = re.compile(r"-->|<--|--+|\*\*|(?<= )[-–—](?= )|(?<=\w)-(?=\s|\*)")


def normalize_for_parse(text: str) -> str:
    """Same-length copy with layout flattened, for the parser only.

    en_core_web_sm reads a newline as an ordinary token boundary, so a wrapped
    line splits a name ("Workvivo\nSupport") while a paragraph break silently
    fuses a heading onto the body. Both are fixed here, character-for-character:

    * a newline inside a sentence becomes a space, joining the wrapped line;
    * a newline that ends a paragraph, list item or already-punctuated line
      becomes ".", giving the parser a real sentence boundary;
    * markdown scaffolding and control characters become spaces;
    * notes punctuation that separates clauses ("-->", " - ", "**") becomes a
      break, since the parser otherwise chunks straight through it.

    The parser never sees the real text, but every offset it returns indexes the
    real text — so nothing here may change the length. The assertion is the
    guard: lose one character and every span downstream points somewhere wrong.
    """
    out = list(text)
    for i, ch in enumerate(out):
        if ch == " ":
            continue
        if ch in _ALWAYS_BLANK or unicodedata.category(ch) in {
            "Cc", "Cf", "Zs", "Zl", "Zp"
        }:
            out[i] = " "

    for match in re.finditer(r"\n", text):
        i = match.start()
        rest = text[i + 1:]
        ends_sentence = text[:i].rstrip().endswith((".", "!", "?", ":", ";"))
        starts_block = bool(re.match(r"[ \t]*(?:\n|[#>*+\-]|\d+[.)])", rest))
        out[i] = "." if (ends_sentence or starts_block or not rest.strip()) else " "

    for match in _LINE_MARKER.finditer(text):
        for i in range(match.start(), match.end()):
            out[i] = " "

    for match in _CLAUSE_BREAK.finditer(text):
        for i in range(match.start(), match.end()):
            out[i] = " "
        out[match.start()] = "."

    normalized = "".join(out)
    assert len(normalized) == len(text), "normalisation must preserve offsets"
    return normalized


def _nlp():
    """Load en_core_web_sm once; importing this module must not load a model."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _NLP


class Span(NamedTuple):
    """One candidate span, addressed by offsets into the clean episode text."""

    sid: str  # "c3" from the parser, "n0" proposed by the model
    start: int
    end: int
    text: str
    candidate_type: str
    origin: str  # "parser" | "model"
    match: str = "exact"  # how a model-proposed span was located


def _is_title_case(text: str) -> str | bool:
    """Every alphabetic word capitalised — "Okta", "Microsoft Entra ID"."""
    words = [w for w in re.findall(r"[^\W\d_]+", text)]
    return bool(words) and all(w[:1].isupper() for w in words)


def classify(text: str, *, is_pronoun: bool, has_proper: bool) -> str:
    """Candidate type from the span's own shape — no model involved.

    Title case backs up the POS tag because en_core_web_sm routinely tags an
    unfamiliar product name as NOUN rather than PROPN, and product names are the
    common case here. A determiner is checked first, so "The company" stays a
    definite description rather than being read as a name.
    """
    if is_pronoun:
        return PRONOUN
    first = text.split(maxsplit=1)[0].lower() if text.split() else ""
    if first in _DEFINITE:
        return DEFINITE_DESCRIPTION
    if has_proper or _is_title_case(text):
        return PROPER_NAME
    return NOMINAL


def parse_spans(text: str) -> list[Span]:
    """Every span the parser can justify, nested spans included.

    Named entities, noun chunks and pronouns, plus proper-noun runs *inside* a
    chunk — "the Neo4j database" must yield "Neo4j" as well, because the two
    refer to different things and only one of them is a graph node.
    """
    doc = _nlp()(normalize_for_parse(text))
    found: dict[tuple[int, int], tuple[str, bool, bool]] = {}

    def add(start: int, end: int, is_pronoun: bool, has_proper: bool) -> None:
        # trim whitespace the normalisation exposed at the span edges
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start or not _HAS_LETTER.search(text[start:end]):
            return  # ":*" and friends are punctuation, not candidates
        prev = found.get((start, end))
        if prev is None:
            found[(start, end)] = (text[start:end], is_pronoun, has_proper)
        else:  # seen from another source — keep the stronger signals
            found[(start, end)] = (prev[0], prev[1] or is_pronoun,
                                   prev[2] or has_proper)

    for ent in doc.ents:
        add(ent.start_char, ent.end_char, False, True)

    for chunk in doc.noun_chunks:
        # en_core_web_sm mis-parses the verb after a name as the chunk head, so
        # "Neo4j stores the graph" yields the chunk "Neo4j stores". A proper name
        # is capitalised; a determinerless chunk that starts capitalised and ends
        # lowercase is a name plus a mis-tagged verb, so keep only the name.
        tokens = list(chunk)
        while tokens and tokens[0].pos_ == "PROPN" and not tokens[0].text[:1].isupper():
            tokens = tokens[1:]
        if (tokens and tokens[0].text[:1].isupper()
                and not any(t.pos_ == "DET" for t in tokens)):
            while len(tokens) > 1 and not tokens[-1].text[:1].isupper():
                tokens = tokens[:-1]
        if not tokens:
            continue
        chunk_start = tokens[0].idx
        chunk_end = tokens[-1].idx + len(tokens[-1])
        has_proper = any(t.pos_ == "PROPN" for t in tokens)
        is_pronoun = len(tokens) == 1 and tokens[0].pos_ == "PRON"
        add(chunk_start, chunk_end, is_pronoun, has_proper)
        # nested proper-noun runs
        run: list = []
        for token in tokens + [None]:
            # en_core_web_sm routinely tags the verb after a name as PROPN
            # ("Neo4j stores", "Okta issues", "syncs team names"). A proper name
            # is capitalised in the source, so a lowercase token ends the run.
            if (token is not None and token.pos_ == "PROPN"
                    and token.text[:1].isupper()):
                run.append(token)
                continue
            if run:
                start, end = run[0].idx, run[-1].idx + len(run[-1])
                if (start, end) != (chunk_start, chunk_end):
                    add(start, end, False, True)
                run = []

    for token in doc:
        if token.pos_ == "PRON":
            add(token.idx, token.idx + len(token), True, False)

    spans = []
    for i, ((start, end), (raw, is_pronoun, has_proper)) in enumerate(
        sorted(found.items())
    ):
        spans.append(Span(
            sid=f"c{i}", start=start, end=end, text=raw,
            candidate_type=classify(raw, is_pronoun=is_pronoun,
                                    has_proper=has_proper),
            origin="parser",
        ))
    return spans


def mark_text(text: str, spans: list[Span]) -> str:
    """Render a marked *copy*: ``⟦c3 the ⟦c4 Neo4j⟧ database⟧``.

    Nesting is preserved so the model can see that a name sits inside a larger
    phrase. The original string is never modified — offsets come from it alone.
    """
    opens: dict[int, list[Span]] = {}
    closes: dict[int, int] = {}
    for span in spans:
        opens.setdefault(span.start, []).append(span)
        closes[span.end] = closes.get(span.end, 0) + 1
    for at in opens:  # longer spans open first so nesting is well formed
        opens[at].sort(key=lambda s: (-s.end, s.sid))

    out: list[str] = []
    for i, char in enumerate(text):
        out.append(_CLOSE * closes.get(i, 0))
        for span in opens.get(i, []):
            out.append(f"{_OPEN}{span.sid} ")
        out.append(char)
    out.append(_CLOSE * closes.get(len(text), 0))
    return "".join(out)


FUZZY_CUTOFF = 0.86
FUZZY_MARGIN = 0.04


def _unique(hits: list[int]) -> int | None:
    return hits[0] if len(hits) == 1 else None


def _find_exact(text: str, needle: str) -> int | None:
    return _unique([m.start() for m in re.finditer(re.escape(needle), text)])


def _find_loose_whitespace(text: str, needle: str) -> tuple[int, int] | None:
    """Match ignoring how whitespace was rendered.

    The common miss by far: the model reads "Workvivo\nSupport" off the marked
    copy and writes back "Workvivo Support". The content is identical, only the
    layout differs, so this is still an exact match — not a guess.
    """
    parts = [re.escape(p) for p in needle.split()]
    if not parts:
        return None
    spans = [(m.start(), m.end())
             for m in re.finditer(r"\s+".join(parts), text)]
    return spans[0] if len(spans) == 1 else None


def _find_fuzzy(
    text: str, needle: str, cutoff: float, margin: float
) -> tuple[int, int, float] | None:
    """Best approximate window, only when it clearly beats every alternative.

    This is the one tier that genuinely guesses, so it is deliberately strict:
    a candidate must clear ``cutoff`` *and* beat the runner-up by ``margin``.
    A near-tie means we cannot tell which occurrence was meant, and picking one
    would defeat the point of answering in ids at all.
    """
    n = len(needle)
    starts = [m.start() for m in re.finditer(r"\S", text)
              if m.start() == 0 or text[m.start() - 1].isspace()]
    scored: list[tuple[float, int, int]] = []
    for start in starts:
        for end in range(max(start + 1, start + int(n * 0.75)),
                         min(len(text), start + int(n * 1.25)) + 1):
            if end < len(text) and not text[end].isspace():
                continue
            ratio = difflib.SequenceMatcher(None, text[start:end], needle).ratio()
            if ratio >= cutoff:
                scored.append((ratio, start, end))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    rival = next((s for s in scored[1:] if not (s[1] < best[2] and best[1] < s[2])),
                 None)
    if rival is not None and best[0] - rival[0] < margin:
        return None
    return best[1], best[2], best[0]


def locate_new_spans(
    text: str,
    proposed: list[dict],
    taken_ids: set[str],
    *,
    fuzzy: bool = True,
    cutoff: float = FUZZY_CUTOFF,
) -> tuple[list[Span], list[dict]]:
    """Re-find model-proposed spans in the clean text, in three tiers.

    exact -> whitespace-insensitive -> fuzzy. Each hit records which tier found
    it, so a fuzzy match is auditable rather than indistinguishable from a
    verbatim one. Every tier still demands a *unique* answer: ambiguity is
    dropped and counted, never resolved by taking the first occurrence.

    The model's own ``n*`` id is preserved — chains reference it, so renumbering
    here would silently break every reference.
    """
    kept: list[Span] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    for item in proposed:
        sid = (item.get("id") or "").strip()
        raw = (item.get("text") or "").strip()
        if not sid.startswith("n") or sid in taken_ids or sid in seen:
            dropped.append({"id": sid, "text": raw, "reason": "bad_or_duplicate_id"})
            continue
        if not raw:
            dropped.append({"id": sid, "text": raw, "reason": "empty"})
            continue

        found, how, score = None, "exact", 1.0
        start = _find_exact(text, raw)
        if start is not None:
            found = (start, start + len(raw))
        else:
            loose = _find_loose_whitespace(text, raw)
            if loose is not None:
                found, how = loose, "whitespace"
            elif fuzzy:
                hit = _find_fuzzy(text, raw, cutoff, FUZZY_MARGIN)
                if hit is not None:
                    found, how, score = (hit[0], hit[1]), "fuzzy", round(hit[2], 3)

        if found is None:
            hits = len(re.findall(re.escape(raw), text))
            dropped.append({
                "id": sid, "text": raw,
                "reason": "not_found" if hits == 0 else f"ambiguous_x{hits}",
            })
            continue

        seen.add(sid)
        span_text = text[found[0]:found[1]]
        kept.append(Span(
            sid=sid, start=found[0], end=found[1], text=span_text,
            candidate_type=item.get("candidate_type") or classify(
                span_text, is_pronoun=False, has_proper=False
            ),
            origin="model",
            match=how if how == "exact" else f"{how}:{score}",
        ))
    return kept, dropped


def merge_spans(
    parser_spans: list[Span], new_spans: list[Span]
) -> tuple[dict[str, Span], list[dict]]:
    """One lookup table over both namespaces; only *crossing* spans collapse.

    Containment is legitimate and must survive: "Neo4j" nested inside "the Neo4j
    database", or "Mean Pooling" inside "Span Mean Pooling", are different
    things, and the nested inventory exists precisely to keep them apart.
    Aliasing a nested span to its container silently merges two entities the
    model deliberately separated.

    A partial overlap — two spans that cross without either containing the other
    — is ill formed, so the longer survives and a parser span wins a tie.
    """
    ordered = sorted(
        parser_spans + new_spans,
        key=lambda s: (-(s.end - s.start), s.origin != "parser", s.sid),
    )
    table: dict[str, Span] = {}
    taken: list[Span] = []
    collapsed: list[dict] = []
    for span in ordered:
        clash = next(
            (t for t in taken
             if span.start < t.end and t.start < span.end          # overlap
             and not (t.start <= span.start and span.end <= t.end)  # not nested
             and not (span.start <= t.start and t.end <= span.end)),
            None,
        )
        if clash is not None:
            collapsed.append({"dropped": span.sid, "kept": clash.sid,
                              "text": span.text})
            table[span.sid] = clash  # alias, so a reference still resolves
            continue
        taken.append(span)
        table[span.sid] = span
    return table, collapsed


def union_identical_names(chains: list[list[Span]]) -> list[list[Span]]:
    """Merge chains that share a proper name with identical text.

    A model can chain three occurrences of a name and miss the fourth, or split
    two identical names into separate chains — that happened live with
    "Span Mean Pooling". Within one episode an identical proper name is a
    deterministic signal, so it does not need to be re-decided by a model.

    Deliberately limited to ``proper_name``. Definite descriptions and nominals
    are context-dependent — "the average" of the whole is not "the average" of
    the span — so those stay the model's call.
    """
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        while parent.setdefault(i, i) != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_name: dict[str, int] = {}
    for i, chain in enumerate(chains):
        find(i)
        for span in chain:
            if span.candidate_type != PROPER_NAME:
                continue
            other = by_name.setdefault(span.text, i)
            parent[find(i)] = find(other)

    grouped: dict[int, list[Span]] = {}
    for i, chain in enumerate(chains):
        grouped.setdefault(find(i), []).extend(chain)

    merged: list[list[Span]] = []
    for members in grouped.values():
        unique = {(s.start, s.end): s for s in members}
        chain = sorted(unique.values(), key=lambda s: s.start)
        head = pick_head(chain)
        merged.append([head] + [s for s in chain if s is not head])
    return merged


def pick_head(chain: list[Span]) -> Span | None:
    """Fallback head when the model's choice did not survive validation.

    Proper name first, then a definite description, then the longest span —
    a chain headed by a pronoun is useless downstream.
    """
    for wanted in (PROPER_NAME, DEFINITE_DESCRIPTION, NOMINAL):
        matches = [s for s in chain if s.candidate_type == wanted]
        if matches:
            return max(matches, key=lambda s: s.end - s.start)
    return max(chain, key=lambda s: s.end - s.start) if chain else None
