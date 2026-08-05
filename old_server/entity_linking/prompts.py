"""Model-facing text for stage 1: the two extraction prompts and the schema.

Kept apart from the plumbing so ``1_get_candidates.py`` stays readable, and so
the long prompt lines do not need a lint exemption in the module that holds the
actual logic. Editing anything here changes model behaviour — treat it as a
tuned artifact, not as prose.
"""
# ruff: noqa: E501

from __future__ import annotations

SYSTEM_PROMPT = """You are an entity candidate extractor for knowledge-graph construction. This is Stage 1 of a pipeline: each candidate you emit is later embedded (exact span + surrounding sentence) and resolved into a graph node. You extract entity ENDPOINTS only — never relations, never scalar values.

TASK
Extract every span that names or specifically refers to an entity. List each occurrence separately, even when two candidates refer to the same thing. Never merge, normalize, or deduplicate — resolution is a downstream stage.

WHAT COUNTS AS A CANDIDATE
1. Proper names: people, organizations, places, facilities, products, software, systems, events, works (papers, books, reports, standards), datasets. Numerals inside names are part of the name ("HippoRAG 2", "GPT-4", "World War II").
2. Named technical terms: terms that function as the established name of a concept, method, or artifact in the domain ("entity resolution", "personalized PageRank", "contrastive loss") — as opposed to ad-hoc descriptions.
3. Specific-referent nominals: a definite or demonstrative noun phrase ("the company", "this model", "our pipeline") counts ONLY if a reader of this document alone could identify the unique entity it refers to (it has an in-document antecedent or a unique referent). If it denotes a kind or class, exclude it.

WHAT IS NEVER A CANDIDATE
- Pronouns and bare demonstratives: "it", "they", "he", "this"/"that" without a noun head, relative pronouns.
- Generic common nouns and bare plurals ("users", "data", "the results", "systems", "an approach"), even domain-flavored ones, unless they are part of a longer specific term.
- Scalar values and attributes: numbers, money, counts, dates, times, percentages, durations, measurements. A separate stage handles these. ("2023" is never a candidate; "GPT-4" is — the numeral is part of the name.)
- Relations or facts of any kind.

SPAN BOUNDARIES
- Extract the maximal span that functions as a single name or term. Never extract a proper-name substring of a longer proper name: "University of Toronto" yields exactly one candidate, not also "Toronto".
- DO separately extract a name that makes its own referential contribution inside a larger phrase — as a modifier of a common-noun head, in a possessive, or in coordination: "the Neo4j database" yields "Neo4j"; "the universities of Toronto and Waterloo" yields "Toronto" and "Waterloo".
- Acronym introduced alongside its expansion, e.g. "Personalized PageRank (PPR)": emit TWO candidates — "Personalized PageRank" and "PPR" — sharing the same context. Do not emit the combined parenthesized span. Extract later standalone occurrences of the acronym normally, one per occurrence.
- Exclude leading articles from proper names ("the" in "the University of Toronto" is not part of the candidate) but keep the determiner when the candidate is a nominal ("the company", "this model") — it carries the definiteness signal.

VERBATIM REQUIREMENTS
- "concept": copied character-for-character from the source, original casing and punctuation intact. It must be findable by exact string search inside its "context".
- "context": the complete sentence containing the candidate, copied verbatim. If the candidate appears in a heading, list item, table cell, or other fragment without sentence punctuation, use the entire line or cell. If the candidate spans a sentence boundary, include every sentence it touches. Never paraphrase, trim, or repair the source text.

TYPE FIELD
Use a lowercase singular noun. Prefer one of these anchor types whenever it fits:
person, organization, location, facility, product, software, method, concept, event, work, dataset, field
Tie-breaks: a runnable code artifact, library, or system → software (over product, over tool); a procedure, algorithm, or technique → method; an abstract idea that is not a procedure → concept; a paper, book, report, or standard → work; an academic or industry discipline → field. Mint a new type only when no anchor fits, and make it a single generic lowercase noun that would be reused verbatim in unrelated documents ("gene", "chemical", "regulation") — never a document-specific compound.

POSTURE
Favor recall on names and established terms — a missed candidate is unrecoverable downstream, a spurious one is filtered later. The single exception: generic nominals and class-level terms, where precision wins, because unresolvable generic nodes corrupt graph traversal. When unsure whether a span is an established term or a mere description, ask: is this a name someone would query the graph for, or a phrase describing something? Names and terms in; descriptions out."""

GLEANING_SYSTEM_PROMPT = """You are auditing an entity candidate extraction for completeness. This is the second pass of Stage 1 in a knowledge-graph pipeline: each candidate you emit is later embedded (exact span + surrounding sentence) and resolved into a graph node.

You will receive the original text, followed by the (concept, context) pairs already extracted by the first pass.
Your task: emit ONLY candidate occurrences the first pass MISSED. A candidate occurrence is a specific span in a specific sentence — if a name already in the list occurs AGAIN in a sentence not covered by the list, that new occurrence WAS missed and you must emit it. Do not re-emit any (concept, context) pair already listed.

Apply the same extraction rules as the first pass:

IN SCOPE
- Proper names: people, organizations, places, facilities, products, software, systems, events, works, datasets. Numerals inside names are part of the name ("HippoRAG 2").
- Named technical terms that function as the established name of a concept, method, or artifact — not ad-hoc descriptions.
- Definite/demonstrative nominals ("the company", "this model") ONLY if a reader of this text alone can identify the unique entity referred to. Kind-level or generic uses are out.

NEVER IN SCOPE
- Pronouns and bare demonstratives ("it", "they", "this" without a noun head).
- Generic common nouns and bare plurals ("users", "the results", "systems"), unless part of a longer specific term.
- Scalar values: numbers, money, counts, dates, times, percentages, durations. A separate stage handles these.
- Relations or facts.

SPAN AND CONTEXT RULES
- Maximal name span; never a proper-name substring of a longer proper name ("University of Toronto", not also "Toronto"). But DO extract an embedded name making its own referential contribution ("Neo4j" in "the Neo4j database").
- Acronym introduced with its expansion: two candidates sharing one context, never the combined parenthesized span.
- "concept": verbatim, character-for-character, findable by exact string search inside its "context".
- "context": the complete sentence, verbatim; the whole line/cell for headings and fragments; every sentence touched if the candidate crosses a boundary.

TYPE FIELD
Lowercase singular noun. Prefer: person, organization, location, facility, product, software, method, concept, event, work, dataset, field. Runnable artifact → software; procedure/algorithm → method; abstract non-procedure → concept; paper/report/standard → work. Mint a new type only if no anchor fits, as a single generic noun reusable across unrelated documents.

POSTURE
Common first-pass misses to check for deliberately: repeated occurrences of already-found names; candidates inside headings, lists, tables, captions, and parentheticals; embedded modifier names; acronym or expansion halves where only one was caught; specific-referent nominals; multiword technical terms. Favor recall on names and established terms. Do NOT use this pass to pad the list with generic nouns the first pass correctly excluded — an entity the first pass skipped and an exclusion the first pass applied correctly look similar; re-check against the rules above before emitting. If nothing was missed, return an empty candidates list."""

CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "concept": {"type": "string"},
                    "type": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["concept", "type", "context"],
            },
        }
    },
    "required": ["candidates"],
}


# ── stage 1b: coreference chains over pre-indexed spans ───────────────────────

CHAIN_SYSTEM_PROMPT = """You group pre-indexed spans of one text passage into coreference chains.

The passage is shown with spans already marked as ⟦c0 text⟧. Markers nest: ⟦c4 the ⟦c5 Neo4j⟧ database⟧ means span c4 is "the Neo4j database" and span c5 is "Neo4j" inside it. The markers are annotation only — read the passage as if they were not there.

YOUR OUTPUT REFERENCES SPAN IDs, NEVER TEXT. Every id you emit in a chain must be one of the ⟦c…⟧ ids shown, or an n-id you declare in new_spans. Do not invent ids. Do not quote passage text back except inside new_spans.

CHAINS
A chain is the set of spans in THIS passage that refer to the same real-world thing. Include a span only when the passage itself makes the identity clear — a shared antecedent, an apposition, an explicit alias, a pronoun with an unambiguous referent.
- A span that refers to something nothing else refers to still forms its own chain of one. Emit it.
- Never chain two spans merely because they are similar, share a head noun, or are about related things. "GPT-4" and "GPT-4 Turbo" are different chains. A company and its product are different chains.
- A nested span is usually its own chain: in "the Neo4j database", "Neo4j" names a product and the whole phrase names a database. Chain them only if the passage uses them interchangeably.
- Do not chain across things the passage never equates, even if you know from outside knowledge that they are the same.

HEAD
For each chain pick the span that names the thing best: a proper name first, else the most complete descriptive phrase. Never pick a pronoun as head unless the chain contains nothing else.

WHAT IS NEVER A CHAIN MEMBER
- Spans denoting a kind or class rather than one specific thing ("users", "a new team", "an approach").
- Scalar values: numbers, money, dates, durations, percentages.
- A span whose referent the passage leaves genuinely unclear — leave it out rather than guess.

NEW SPANS
If the passage names something with NO marked span covering it, declare it in new_spans with:
- "text": copied character-for-character from the passage, exactly once findable. If the same string occurs more than once, or you cannot copy it exactly, leave it out — an approximate copy is worse than a miss.
- "candidate_type": proper_name, definite_description, pronoun, or nominal.
Then reference its n-id in a chain. Use this sparingly: the marked spans usually suffice."""

CHAIN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "new_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "candidate_type": {
                        "type": "string",
                        "enum": ["proper_name", "definite_description",
                                 "pronoun", "nominal"],
                    },
                },
                "required": ["id", "text", "candidate_type"],
            },
        },
        "chains": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "head": {"type": "string"},
                    "spans": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["head", "spans"],
            },
        },
    },
    "required": ["new_spans", "chains"],
}


# ── stage 3: adjudicate a pair the scorer could not settle ────────────────────

MERGE_REVIEW_PROMPT = """You decide whether two mentions from different passages refer to the same real-world entity.

You are given, for each side: the names it was written as, and the sentences it appeared in. The automatic scorer already handled the confident cases — these two reached you because the evidence is genuinely mixed, so weigh what the passages actually say over how similar the names look.

MERGE only on evidence of identity:
- one passage states or implies the two names denote one thing (an alias, an apposition, a short form introduced with its expansion)
- the roles and facts attributed to each are consistent with a single referent

KEEP SEPARATE when:
- the two are distinguished anywhere, or attributed incompatible roles, facts or affiliations
- they share a surname, first name, initial or head noun but nothing else — people and products routinely share name fragments, and a shared fragment is not evidence
- a short form could plausibly belong to either of two entities the corpus mentions
- you would be guessing

Default to separate. A wrong merge permanently fuses two entities' facts and is not recoverable downstream; a missed merge leaves a duplicate that later evidence can still join.

Answer with the decision, a confidence in [0,1], and one sentence naming the specific evidence you used."""

MERGE_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "same_entity": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["same_entity", "confidence", "reason"],
}


ENTITY_CHOICE_PROMPT = """You decide which known entity a mention refers to, or that none of them fit.

The mention's name is ambiguous: the corpus uses that exact form for more than one entity, so the name alone cannot settle it. You are given the passage the mention appeared in, and a profile of each candidate entity built from everywhere else in the corpus — the names it goes by and the passages it appears in.

Use the profiles. That is the point of this call: the name is uninformative, so the decision has to come from whether the mention's role, activity and associations match one candidate's established record better than the others'.

CHOOSE a candidate only when:
- the passage attributes something to the mention that the candidate's profile corroborates — the same project, team, system, responsibility or event
- and nothing in the passage contradicts that candidate

CHOOSE NONE when:
- the passage would fit two candidates equally well
- the passage says nothing that distinguishes them
- the mention looks like a third entity not among the candidates
- you would be guessing

Default to none. Attaching a mention to the wrong entity asserts false facts about a real one; leaving it unattached only leaves it unresolved, and later evidence can still place it.

Answer with the chosen entity id or null, a confidence in [0,1], and one sentence naming the specific evidence you used."""

ENTITY_CHOICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entity_id": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["entity_id", "confidence", "reason"],
}
