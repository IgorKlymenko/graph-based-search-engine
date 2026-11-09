# Ideas

Product ideas before they are proposals. A row in [PROPOSALS.md](PROPOSALS.md)
has a scope, a size and a place in the code; an idea here only has to be worth
keeping. When an idea gets measured or scoped, it moves there and this entry
points at the row. Keep the *why* and the *signals we already have*: those are
what make an idea cheap to pick up later.

Model-facing like the rest of the wiki: facts and opinions kept apart, dates
absolute, nothing here is a commitment.

---

## 1. When unsure, point to the person who would know (2026-09-21)

**The idea.** Today the answer step has three stances: answer, reframe the
question, or ask which entity was meant (ADR slack 16). Add a fourth outcome
for low confidence: *refer*. Instead of a hedged answer or "not in the notes",
Nazar names the person most likely to know and says why:

> I can't find this written down. @Arsenii would know: he decided the Neo4j →
> Supabase move in #eng on 2026-05-03 and wrote the migration.

In some cases that is the whole answer: the contact of the best person to ask.
The reason matters as much as the name, because it lets the asker judge the
referral and gives the referred person context when they are pinged.

**Who counts as "would know", in order of strength.**

| Signal | Where it already is, or what it needs |
|---|---|
| **Made the decision.** Authored the Slack message that settled it, or the ADR entry | Slack messages carry `user`; thread documents keep authors per message (`slack.py` renders `display_name`). ADRs have no author field; git history has one (`git log` on `ADRs/`), which the GitHub connector could record on the document. |
| **Wrote the solution.** Author of the code the question is about | Not stored. The GitHub connector fetches blobs by sha; adding `git log -1 --format=%an` per path (local) or the commits API (remote) to `files` is one column. Blame per symbol is more precise and more expensive. |
| **Was in the room.** Participated in the thread or meeting where it came up | Thread participants are in the thread document; meeting notes are a future connector (calendar attendees, transcript speakers). |
| **Talks about it a lot.** Co-occurs with the question's entities across the corpus | Free from the graph: people are entities too (`slack_users`, mentions), so PPR from the question's seeds already ranks people by proximity. The strongest general signal and the cheapest to try. |
| **Has the role.** Their title concerns the topic (security questions → whoever is "Security" or "Infra") | Slack profile `title` per user; a small role → topic mapping, or an LLM judgement over the title. Weakest and most likely to be wrong, so it should only break ties. |

**Where it hangs in the code.** `answer.ANSWER_SCHEMA` gains `refer:
{person, because}`; `retrieve` returns, beside passages, the top people by the
walk and by document authorship of the passages it did find; the stance model
picks `refer` when the passages do not answer and a person scores clearly above
the rest. The Slack renderer mentions the person only if they are in the
workspace and have not opted out.

**Why it is worth doing.** The honest failure mode of any knowledge base is
"it isn't written down", and in a small company the answer is nearly always a
person. A referral turns a dead end into the next step, and the reason ("he
decided it in #eng") is exactly the provenance the graph is built to hold.
Measured on the vault-plus-code tenant, the answer step said the passages
lack the information for 4 of 28 hand-written questions and 22 of 116 with
file tools alone in the agent rig ([experiments.md](experiments.md) § Agents
and tokens); each of those is a candidate referral.

**What could go wrong.**

- Pointing at the wrong person costs two people's time and Nazar's credit; the
  reason line and a confidence floor are the guard, and a referral should never
  be the *only* thing tried: reframe and clarify come first.
- Naming someone reveals where they were and what they wrote. Only people in
  the asker's workspace, only from channels the asker can read, never from DMs;
  a per-person opt-out; and no referral for questions the tenant's admin marks
  sensitive.
- The same expert gets pinged for everything. Rate-limit referrals per person
  per day and prefer a document when one exists at all.

**How to know it works.** Log every referral with its reason. A referral
*hits* when the referred person replies in that thread within a day, or the
asker reacts to it. Report hit rate per signal type; drop the signals that do
not hit.

**Smallest version to try.** People as entities plus the walk: for a question
with no answering passage, rank people by PPR proximity to the seeds and show
the top one with the passages that put them there ("mentioned with Supabase
and RLS in 14 messages"). No new data, one new stance, one renderer branch.
Measure hit rate for two weeks before adding authorship.
