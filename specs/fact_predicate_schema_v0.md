# Fact predicate schema — starter set (v0, worth testing now)

## Status: hypothesis, not truth
This is a **query-driven starter set**, small on purpose. It is a hypothesis about which
facts your queries will filter on — correct it from two sources of evidence:
1. Your real query list (what you'd demo to a customer). A predicate earns a place here only
   if a query will **filter, aggregate, or group** on it.
2. What accumulates in the `other` bucket during test runs. Frequent `other` predicates are
   your signal to promote a new one into the closed set.

Test **Tier A first** (highest frequency in your corpus); add Tier B only when a query needs it.

## Two predicate families (the structural split)
Every fact is one of two shapes. The family determines how the value is handled.

- **Attribute fact** — subject entity + predicate + **scalar typed value** (money / count / date /
  percentage / text-category). The value goes through the deterministic normalizer.
  Example: `(TeamX, headcount, 12)`.
- **Relational fact** — subject entity + predicate + **object entity**. The "value" is another
  node; it resolves to a canonical entity ID via your resolution path. No scalar parsing.
  Example: `(TeamX, usesTool, SharePoint)` where SharePoint is an entity node.

Both are reified Fact nodes, both carry bi-temporal validity + provenance + confidence. Only
the value slot differs: attribute → typed scalar; relational → `object_entity_id`.

## value_type → handler (attribute facts)
| value_type   | handler / library                              | typed output                          |
|--------------|------------------------------------------------|---------------------------------------|
| `money`      | price-parser → Decimal; babel currency precision | `amount_minor: int`, `currency: str` (ISO 4217). **Never float.** |
| `count`      | int parse; strip approximators ("~","approx")  | `count: int`, `approximate: bool`     |
| `percentage` | numeric parse                                  | `basis_points: int` (exact)           |
| `date`       | dateparser / python-dateutil                   | `iso_date: str` (ISO 8601)            |
| `duration`   | dateparser / dateutil                          | `iso_duration: str` (ISO 8601)        |
| `text`       | none — store verbatim (for categories/qualitative) | `text: str`                       |

Relational facts skip this table; their object resolves through entity resolution to
`object_entity_id`.

## The closed predicate set

### Tier A — test now
| predicate       | family     | value_type  | meaning / when to use                                                        | borrowed from |
|-----------------|------------|-------------|------------------------------------------------------------------------------|---------------|
| `usesTool`      | relational | entity_ref  | subject (team/person/process) uses tool or system X                          | Wikidata `uses` (P2283); no clean schema.org term |
| `ownerOf`       | relational | entity_ref  | subject is accountable for X (process, tool, budget, domain)                 | ~schema.org `owns` / Wikidata P1830 — VERIFY before pinning a URL |
| `hasRole`       | attribute  | text        | subject holds role/title X. **Role stays a string, not an entity node.**     | schema.org `jobTitle` |
| `worksFor`      | relational | entity_ref  | person → org or team they belong to                                          | schema.org `worksFor` (confident) |
| `documentedIn`  | relational | entity_ref  | knowledge/process X lives in source Y (doc, wiki, video). NOT this fact's provenance — this is a queryable "where do I find X" fact | domain-specific — name your own |
| `headcount`     | attribute  | count       | size of an org or team (unit: people)                                        | schema.org `numberOfEmployees` (confident) |

### Tier B — add when a query demands it
| predicate        | family     | value_type   | meaning / when to use                                        | borrowed from |
|------------------|------------|--------------|-------------------------------------------------------------|---------------|
| `responsibleFor` | relational | entity_ref   | subject leads or performs X (project, task)                 | domain-specific |
| `partOf`         | relational | entity_ref   | team/process is part of a larger org or department          | schema.org `parentOrganization`/`memberOf` — VERIFY |
| `foundingDate`   | attribute  | date         | when an org was founded                                     | schema.org `foundingDate` (confident) |
| `revenue`        | attribute  | money        | org revenue                                                 | no clean schema.org term; cf. Wikidata P2139 — name your own |
| `fundingRaised`  | attribute  | money        | amount an org has raised                                    | domain-specific |
| `fundingStage`   | attribute  | text         | seed / Series A / etc. (consider a fixed sub-enum)          | domain-specific |
| `industry`       | attribute  | text         | org vertical (fintech, biotech, …)                          | schema.org `industry` — VERIFY / name your own |
| `changedOn`      | attribute  | date         | a stated event/change date; feeds a fact's valid-time       | domain-specific |

### Escape hatch (always present)
| predicate | family | value_type | meaning |
|-----------|--------|------------|---------|
| `other`   | —      | text       | no predicate fits. Model MUST fill `predicate_raw` with what it wanted. Never force-fit; never drop. This bucket is your evidence source for what to promote. |

## Enum for structured output (paste into your JSON schema)
```json
"predicate": {
  "type": "string",
  "enum": [
    "usesTool", "ownerOf", "hasRole", "worksFor", "documentedIn", "headcount",
    "responsibleFor", "partOf", "foundingDate", "revenue", "fundingRaised",
    "fundingStage", "industry", "changedOn",
    "other"
  ]
}
```
Pair this with `predicate_raw` (string, required only meaningfully when predicate is `other`),
and glosses in the system prompt so the model can choose correctly within the set. The enum
enforces membership; the glosses guide the choice; verify the choice against the value_type
contract downstream (an in-set predicate can still be the wrong one).

## Pydantic (v2)
```python
from enum import Enum

class Predicate(str, Enum):
    uses_tool       = "usesTool"
    owner_of        = "ownerOf"
    has_role        = "hasRole"
    works_for       = "worksFor"
    documented_in   = "documentedIn"
    headcount       = "headcount"
    responsible_for = "responsibleFor"
    part_of         = "partOf"
    founding_date   = "foundingDate"
    revenue         = "revenue"
    funding_raised  = "fundingRaised"
    funding_stage   = "fundingStage"
    industry        = "industry"
    changed_on      = "changedOn"
    other           = "other"

# predicate -> (family, value_type) contract, used to route normalization and to VALIDATE
PREDICATE_CONTRACT = {
    "usesTool":      ("relational", "entity_ref"),
    "ownerOf":       ("relational", "entity_ref"),
    "hasRole":       ("attribute",  "text"),
    "worksFor":      ("relational", "entity_ref"),
    "documentedIn":  ("relational", "entity_ref"),
    "headcount":     ("attribute",  "count"),
    "responsibleFor":("relational", "entity_ref"),
    "partOf":        ("relational", "entity_ref"),
    "foundingDate":  ("attribute",  "date"),
    "revenue":       ("attribute",  "money"),
    "fundingRaised": ("attribute",  "money"),
    "fundingStage":  ("attribute",  "text"),
    "industry":      ("attribute",  "text"),
    "changedOn":     ("attribute",  "date"),
    "other":         (None,         "text"),
}
```

## Notes / limits
- **Verify the borrowed identifiers before pinning URLs.** Confident: `worksFor`, `jobTitle`,
  `numberOfEmployees`, `foundingDate`. Uncertain (marked VERIFY): the rest — several of your
  most useful predicates (`usesTool`, `ownerOf`, `documentedIn`) are domain-specific and have
  no clean standard term, which is expected. Name those yourself.
- **The role decision is baked in:** `hasRole` is an attribute with a `text` value, NOT a
  relational link to a role entity — consistent with roles being excluded from entity nodes.
  If you later decide roles ARE nodes, `hasRole` moves to relational/entity_ref.
- **`fundingStage` / `industry` are categoricals.** If your queries group on them, promote
  each to its own fixed sub-enum rather than free text, so grouping is exact.
- Keep this set small. Resist adding a predicate for a fact no current query filters on —
  that is maintenance cost with no payoff. Grow from `other`, on evidence.
