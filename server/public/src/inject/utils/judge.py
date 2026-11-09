"""Which surfaces are names. One short LLM call per forty new surfaces.

Measured on the author's vault against 410 hand labels (wiki/experiments.md
§ Junk filtering): about 60% of the surfaces the pipeline stores are not
entities; no cheap rule separates names from descriptions (best F1 0.445); a
rule inside the coref prompt is ignored; a judge shown the surface in its own
context reaches F1 0.80 with gpt-4.1 and 0.53 with gpt-4.1-mini, which keeps
everything gold keeps and twice as much junk. That recall is what makes the
cascade work: the cheap model goes first and drops what it is sure of, the
strong model sees the 44% that survive -- F1 0.82 at about 60% of the cost of
the strong model alone. Verdicts are cached in ``store["names"]`` by surface,
so a re-run costs nothing and a sync judges only what is new.

Consumers: stage 3 drops non-name surfaces from a chain's identity and
withholds chains left with none (no entity, ``entity_id`` NULL, as for
contested names); stage 5 writes the flag to ``surfaces.is_name``; seeding
and embedding skip non-names. A surface with no verdict is treated as a name.

Editing ``JUDGE_PROMPT`` changes behaviour; treat it as a tuned artifact and
re-score with ``old_server/entity_linking/demos/eval_junk_gold.py``.
"""

from __future__ import annotations

from collections.abc import Callable

from src.inject._stages import load

CH = load("2_chains", "src.inject.chains")

# Cheapest first. A model that drops a surface is final; only survivors reach
# the next one. One model means no cascade.
JUDGE_MODELS: tuple[str, ...] = ("gpt-4.1-mini", "gpt-4.1")
BATCH = 40
WORKERS = 8
CONTEXT_CHARS = 120
CHECKPOINT_BATCHES = 10  # ~400 verdicts between saves

JUDGE_PROMPT = """You decide whether a short span from someone's working notes is a NAME worth an entity node in a personal knowledge graph. The test: would the author look this thing up by this name, and does the span denote exactly one specific thing?

KEEP
- proper names: people, organisations, products, tools, libraries, services, places, works (books, papers, named notes), languages, events
- established technical terms and concepts (self-attention, primary key, temporal validity), including identifiers from the author's own code or schema (file_id, app_user, HAS_FACT, max_tokens), and typo'd or determiner-prefixed forms of any of these ("the supbase", "the RLS", "the CSR")

DROP
- generic nouns and noun phrases, however definite: "the model", "the results", "the data model", "the clustering", "this company", "my research", "the tenant"
- descriptions of things rather than their names; indefinite phrases ("a policy", "a session") unless the head noun is itself an established term
- section headings and list labels that are ordinary words ("Ideas", "Approaches", "Keys", "Discipline", "Evidence", "Confidence")
- adjectives, verbs, keywords and code fragments (ONLY, NEVER, COUNT, MATCH, "DESC;"), spans broken by stray punctuation, spans that run across a line break or a bullet
- several names in one span ("Thomson Reuters / LexisNexis", "Tettra / Guru", "Azure monitor + Cost Management")
- numbers, dates, durations, prices, percentages ("87 days", "2 weeks", "0.09/Gig/Hour")
- pronouns, determiners, quantifiers ("anything", "everyone")
- sentence fragments in any language that are not names

Judge each item on its own; the context is only there to show what the span refers to. When unsure, DROP: a missed name can be added later, a junk entity poisons retrieval.

NORMALISE FIRST
Strip a leading article or possessive (the, a, an, this, that, my, our, your, their) and any markup characters (==, **, backticks, stray quotes) and judge what is left: "the TLS" -> TLS -> KEEP; "the model" -> model -> DROP; "==data monetization" -> data monetization -> KEEP (an established topic); "a nearest-centroid classifier" -> nearest-centroid classifier -> KEEP (a named method). Tools and formats written lowercase or as extensions are names: "uv", "git", ".parquet".

CAPITALISATION IS NOT A NAME
A capitalised word or phrase that is a heading, bullet label, adjective or section title ("Custom", "Flexible", "Brain", "Social Events", "Bonus Build", "Direct Mentorship", "Agentic", "Powers Wikipedia") is DROP unless it is a proper name or established term you would expect to find in a reference work."""

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"id": {"type": "string"}, "keep": {"type": "boolean"}},
                "required": ["id", "keep"],
            },
        }
    },
    "required": ["verdicts"],
}

Completion = Callable[..., dict]


def contexts_for(store: dict, surfaces: set[str]) -> dict[str, str]:
    """One context window per surface: its first live occurrence, ±CONTEXT_CHARS.

    Candidates carry offsets into content-addressed episode text, so the window
    is exact. A surface whose every occurrence is in a retired episode gets no
    context and is not judged this run.
    """
    out: dict[str, str] = {}
    episodes = store["episodes"]
    for candidate in store["candidates"].values():
        surface = candidate["text"]
        if surface not in surfaces or surface in out:
            continue
        episode = episodes.get(candidate["episode_id"])
        if episode is None or episode.get("removed_at"):
            continue
        text = episode["text"]
        lo = max(0, candidate["start"] - CONTEXT_CHARS)
        hi = min(len(text), candidate["end"] + CONTEXT_CHARS)
        out[surface] = text[lo:hi].replace("\n", " ")
    return out


def judge_surfaces(
    contexts: dict[str, str],
    *,
    models: tuple[str, ...] = JUDGE_MODELS,
    workers: int = WORKERS,
    complete: Completion | None = None,
    on_progress: Callable[[dict[str, bool]], object] | None = None,
) -> dict[str, bool]:
    """``surface -> is it a name`` for every surface some model answered.

    Models run as a cascade, cheapest first: a surface a model drops is
    finished, a surface it keeps goes on to the next model, and the last
    model's verdict stands. Within a model, batches of ``BATCH`` run in
    parallel; a batch can come back with mangled ids and lose every item in
    it, so what is still missing is retried in eights and then one at a time.
    A surface no model answers is left out and treated as a name by callers.
    ``on_progress`` receives the verdicts settled so far every ``CHECKPOINT_BATCHES``
    batches, so a caller can persist them before the stage ends.
    """
    complete = CH._complete_json if complete is None else complete
    verdicts: dict[str, bool] = {}
    alive = dict(contexts)
    for model in models:

        def settled(
            partial: dict[str, bool], _alive=alive, _last=(model == models[-1])
        ) -> None:
            if on_progress is None:
                return
            drops = {s: False for s, keep in partial.items() if not keep}
            keeps = {s: True for s, keep in partial.items() if keep} if _last else {}
            on_progress({**verdicts, **drops, **keeps})

        answered = _judge_with(
            alive, model=model, workers=workers, complete=complete, on_batch=settled
        )
        for surface, keep in answered.items():
            if not keep:
                verdicts[surface] = False
        alive = {s: c for s, c in alive.items() if answered.get(s, True)}
        if model == models[-1]:
            verdicts.update({s: True for s in alive if s in answered})
    return verdicts


def _judge_with(
    contexts: dict[str, str],
    *,
    model: str,
    workers: int,
    complete: Completion,
    on_batch: Callable[[dict[str, bool]], object] | None = None,
) -> dict[str, bool]:
    """One model over ``contexts``; missing keys are the surfaces it never answered."""
    items = [(f"s{i}", s, ctx) for i, (s, ctx) in enumerate(sorted(contexts.items()))]
    verdicts: dict[str, bool] = {}

    def ask(batch: list[tuple[str, str, str]]) -> dict[str, bool]:
        user = "\n".join(
            f"{k} | {s.replace(chr(10), ' ')} | {ctx}" for k, s, ctx in batch
        )
        answer = complete(
            system=JUDGE_PROMPT,
            user=user,
            model=model,
            schema=JUDGE_SCHEMA,
            schema_name="name_verdicts",
        )
        by_id = {k: s for k, s, _ in batch}
        return {
            by_id[v["id"]]: bool(v["keep"])
            for v in answer.get("verdicts", [])
            if v.get("id") in by_id
        }

    for size in (BATCH, 8, 1):
        todo = [it for it in items if it[1] not in verdicts]
        if not todo:
            break
        batches = [todo[i : i + size] for i in range(0, len(todo), size)]
        for n, (_, part) in enumerate(
            CH.completed(ask, batches, workers=workers, label=f"{model} judge batches"),
            start=1,
        ):
            verdicts.update(part)
            if on_batch and n % CHECKPOINT_BATCHES == 0:
                on_batch(verdicts)
    return verdicts
