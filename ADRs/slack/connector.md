# Decision Log — Slack connector

Decisions taken while designing the Slack app: how a workspace becomes a tenant,
what a Slack "document" is, when the bot speaks, and what it is structurally
incapable of reading.

Companion to `../supabase/security.md`, which covers roles and tenant isolation.
Entries 1, 4, 8 and 9 there constrain this design and are not re-argued here.

Each entry: what was decided, why, and what it costs. Supersede rather than edit
once shipped.

Status key: **Accepted** (decided and being built) · **Deferred** (decided not
to build yet) · **Open** (identified, not yet decided) · **Superseded**

Claims marked *measured* were verified against this repository. Claims about
Slack's own limits are marked *unverified* and must be checked against current
API docs before they are relied on — they are policy, and policy moves.

---

## Installation

### 1. Install auto-provisions the tenant, the subscription and the installation — *Accepted*

**Context.** The user finds the app in the Slack Marketplace, adds it from their
workspace, and grants scopes. There is no prior account: the workspace *is* the
signup.

**Decision.** The OAuth callback performs one transaction creating `tenants`,
`subscriptions` (trialing) and `connector_installations`. No operator step, no
manual token handling.

```sql
INSERT INTO tenants (id, name) VALUES ($tenant, $team_name);
INSERT INTO subscriptions (tenant_id, sub_status) VALUES ($tenant, 'trialing');
INSERT INTO connector_installations (source, external_id, tenant_id, token_ref)
VALUES ('slack', $team_id, $tenant, $secret_handle)
ON CONFLICT (source, external_id) DO UPDATE
   SET token_ref = EXCLUDED.token_ref, uninstalled_at = NULL;
```

**Why.** Signup friction is the whole funnel. It also means there is exactly one
code path that can mint a tenant, which is the path a security review looks at.

**Cost.** A workspace that installs, uninstalls and reinstalls must not
accumulate tenants — see 2. And an abandoned trial leaves a tenant with a
corpus; retention is unspecified (Open item A).

### 2. Reinstall reuses the tenant — *Accepted*

**Decision.** `ON CONFLICT (source, external_id) DO UPDATE` on the installation
row, keyed on the Slack `team_id`. The tenant is never re-minted.

**Why.** The primary key is `(source, external_id)` precisely so a workspace has
one identity. Minting a second tenant on reinstall would strand the entire
indexed corpus behind an id nothing points at any more, with no error and no
symptom other than an empty product.

**Cost.** A workspace cannot deliberately start fresh without an explicit reset
path, which does not exist yet.

### 3. The token never lands in Postgres — *Accepted, inherited*

`oauth.v2.access` returns the bot token; the callback writes it to the secret
store and persists only the handle in `token_ref`. This restates entry 9 of
`../supabase/security.md` and is repeated here only because the automated
install path is where it would most plausibly be got wrong.

---

## What the app can read

### 4. "Never DMs" is a scope decision, not a runtime check — *Accepted*

**Decision.** Do not request `im:history` or `mpim:history`. Request
`channels:read`, `channels:history`, `app_mentions:read`, `chat:write`,
`users:read`, `files:read`.

**Why.** A rule enforced in code can be defeated by a bug. A scope never granted
means Slack does not deliver the events at all — the guarantee holds even if the
ingest path is wrong. It is also the version that can be stated honestly on a
marketplace listing.

**Cost.** None, unless a later feature genuinely needs DMs, at which point it is
a re-consent, not a config change. A `channel_type` check on ingest stays as
belt-and-braces.

### 5. Private channels are out of scope for v1 — *Deferred*

**Context.** `groups:history` covers private channels. "All group chats" is
ambiguous between public and private.

**Decision.** Public channels only at launch.

**Why.** Private-channel content carries a materially different consent
expectation, and the current isolation model is tenant-level: anyone who can
query the tenant can reach anything indexed in it. Indexing a private channel
would make its contents answerable to workspace members who were never in it.

**Cost.** The most valuable content is often in private channels. Revisit
together with Open item B (per-channel ACL).

---

## What a document is

### 6. A thread is a file — *Accepted*

**Decision.**

```
external_id   thread/{channel_id}/{thread_ts}
title         first message, first line, truncated
url           chat.getPermalink on the parent message
```

**Why.** A thread is the smallest self-contained unit of Slack discourse: it has
a topic, participants and a resolution. `{channel}/{thread_ts}` is unique
workspace-wide and stable across edits, so it is a sound natural key for
`(source, external_id)`.

**Cost.** Threads are append-only and immortal, so a file is never "finished" —
see 8.

### 7. Loose messages bundle into channel-day files — *Accepted*

**Decision.** Top-level messages with no replies are rendered into
`day/{channel_id}/{YYYY-MM-DD}`.

**Why.** Most Slack messages are never replied to. Under 6 alone, every
*"anyone know why staging is down"* would become its own file, episode and
extraction call — the majority of spend on the least useful content.

**Cost.** A day file changes each time the day receives a message, re-rendering
until the day closes. By 8 that costs one episode per change, not the file.

**Open sub-decision.** A message that starts unthreaded and later receives a
reply is indexed in both the day file and its new thread file. Proposed rule: a
message belongs to the day file *only while it has no replies*, and promotion
re-renders that day. Undecided; if left unresolved the content is silently
duplicated and the only symptom is subtly inflated co-occurrence.

### 8. One message per paragraph, so appends are cheap — *Accepted, measured*

**Decision.** Render each Slack message as its own blank-line-separated
paragraph:

```
[14:02] Sarah Chen: the Okta rollback failed again

[14:05] Steven Chen: same error as Tuesday?
```

**Why.** `split_episodes` (`old_server/entity_linking/1_episodes.py`) packs
paragraphs greedily left to right, and `episode_id = uuid5(file_id|sha1(text))`.
Appending therefore leaves every earlier episode byte-identical, so a reply
arriving a week later re-extracts **one** episode rather than the thread. This
is what makes an append-only source affordable at all.

**Cost.** A mid-thread *edit* changes that paragraph's length, shifts all
downstream packing, and re-extracts the tail. Accepted: edits are far rarer than
appends.

### 9. Mentions and markup are resolved before hashing — *Accepted*

**Decision.** `<@U0123>` → display name, `<http://x|text>` → `text`,
`<#C123|eng>` → `#eng`, `subtype` messages (`channel_join`, `bot_message`)
dropped. All of it before the document is hashed or split, in a
`render_slack_thread()` mirroring `strip_markup()`.

**Why.** The extraction prompt requires every context to be findable by exact
string search inside its episode. Resolving after storage breaks that invariant,
and the failure is silent — contexts stop matching and extraction degrades with
no error. The same reasoning produced `strip_markup` for Obsidian.

**Cost.** Display names are resolved at render time, so a later rename does not
propagate to already-indexed text. Acceptable; the alternative is re-rendering
the corpus on every rename.

### 10. Attachments are their own files, not part of the thread — *Accepted*

**Decision.** A Slack file becomes a `files` row keyed on Slack's `file_id`.

**Why.** The same PDF shared in three channels is one document, and
`(source, external_id)` gives that dedupe for free. `render_version` already
exists for exactly this: improve PDF extraction, bump it, re-render the file
without touching thread episodes.

**Cost.** The link between a discussion and its attachment is not modelled. Co-occurrence
recovers most of it in practice, since thread and file share entities — but that
is a reliance, not a guarantee.

**Note.** `url_private_download` requires `Authorization: Bearer <token>`; an
unauthenticated GET returns an HTML login page with status 200, which naive code
stores as document text. External stubs (Drive, Dropbox) carry `external_type`
and have no downloadable content — record the reference, let a future connector
supply the text.

---

## Backfill

### 11. Indexing is progressive, not a blocking wait — *Accepted*

**Context.** The intended UX was "wait ~20 minutes while we process everything".

**Decision.** Index recent history first (proposed: 30–60 days), make the
workspace answerable as soon as that lands, backfill older history in the
background, and notify the installer when it completes.

**Why.** Two independent limits stack, and neither is under our control.
*(unverified)* Slack throttles `conversations.history` and
`conversations.replies` far harder for non-Marketplace apps than for approved
ones — reportedly ~1 request/minute — which makes a full backfill slow before
approval and is a chicken-and-egg problem with getting approved. Separately, the
extraction pipeline itself ran 253 episodes in roughly ten minutes *(measured,
Obsidian corpus)*; a real workspace is thousands of episodes.

**Cost.** More states to represent (partially indexed) and a notification path.
In exchange the promise is one that can actually be kept.

### 12. A `connector_channels` table is required — *Accepted*

**Decision.** Add roughly:

```
connector_channels(source, external_id, channel_id, opted_in,
                   last_cursor, backfilled_through, ...)
```

**Why.** Per-channel opt-in state and a resumable backfill cursor have nowhere
to live today. Without the cursor an interrupted backfill restarts.

**Cost.** One migration. Shape not yet settled — Open item C.

---

## When the bot speaks

### 13. Proactive answering is opt-in per channel, and ships silent first — *Accepted*

**Context.** The intent is that any message is classified, and answered if it is
a question the corpus can address.

**Decision.** Launch responding only to `@mention` and a slash command. Proactive
answering is enabled per channel, and when first enabled it runs in **silent
mode**: classify and log what it *would* have said, post nothing.

**Why.** A bot that interjects uninvited is the standard way an app gets
uninstalled, independent of answer quality — one bad read of "is this for me" in
a busy channel is enough. Silent mode produces the precision number that decides
whether proactive is viable, on real traffic, at no risk.

**Cost.** The headline feature is off by default. Deliberate: it can be turned on
with evidence instead of hope.

### 14. Heuristics gate the classifier, not the other way round — *Accepted*

**Decision.** Cheap filters first (contains `?`, question words, minimum length,
not a bot, not a `subtype`), then a small-model classifier, then retrieval.

**Why.** An LLM call per message in a busy workspace is mostly spend on "lol",
"+1" and deploy notifications. The prefilter removes the great majority before
anything is paid for.

**Cost.** Questions phrased without a question mark are missed. Tunable, and
measurable directly from silent mode.

### 15. Ambiguity is answered with a question, in thread — *Accepted*

**Decision.** When a mention is too vague to answer, reply in-thread asking for
clarification, keyed on `thread_ts`.

**Why.** The retrieval path already produces the signal: if PPR seeds come back
flat and low-scoring, nothing in the corpus is a confident match. That is a
better trigger than asking a model whether the question was clear.

**Cost.** Needs conversation state across two events. `internal.events` carries
dedupe and the worker claim; whether `thread_ts` state belongs there or in its
own table is undecided.

### 16. The answer may say the question is the wrong one — *Accepted, measured*

**Context.** Retrieval collects what the notes say about the names a question
uses. Sometimes that material shows the question itself is off: it assumes a
tool the notes record as dropped, a plan that was superseded, a provisional
choice presented as final, or a means where the notes reached the goal another
way. Answering literally is the classic failure of a retrieval system —
confident, cited, beside the point.

**Decision.** The answer step (`old_server/entity_linking/answer.py`) decides a
*stance* before answering, as structured output: `answer`, `reframe` (one
sentence on why the question is off, the better question, the note titles that
show it, then the answer to the better question) or `clarify` (a name in the
question denotes several entities — the fan-out `seed_entities` now reports —
so ask which). A reframe must cite passages; a missing answer is never grounds
for one. `--no-reframe` pins the stance to `answer`.

**Measured.** Twelve questions over the author's vault, passages selected by
keyword, `gpt-4.1`, three runs each: fine questions left alone **7/7**, bad
premises caught **3/5**, stance flipped across runs **0/12**. The two misses
are borderline (the notes do record running the MCP server locally once; the
"always-up database" answer reframes in substance while labelling itself an
answer). Prompt v2 added the "provisional choice presented as settled" ground
after v1 missed "why did we pick HDBSCAN as the final approach".

**Cost.** One structured completion, as before. The risk is nagging: a bot that
disputes questions is uninstalled as fast as one that interjects (entry 13), so
the prompt defaults to `answer` and the controls are the number to watch.

**Amended (2026-09-19).** Replies are Block Kit: preface section, answer section, sources as a context line with readable titles (linked when the file has a URL); the prompt carries Slack format rules (lead with the answer, under 120 words, plain sentences, no Markdown Slack cannot render, no restating the preface). Re-scored on the twelve questions after the change; see `wiki/experiments.md`.

**Amended (same day as the decision).** Built into the ingress: `handle_event` answers an
`@mention` through `src/query/retrieve.answer_question` as `app_connector`
for the event's `team_id`, and `answer.render_slack` posts a reframe as a
two-line preface (*Wrong question, per the notes:* … / *Better question:* …)
above the answer, a clarification as bullets. Silent-mode logging of would-be
reframes is still to do.

### 17. A thread Nazar has answered in is a conversation — *Accepted*

**Context.** A first answer is rarely the end: "and the pool model?", "why?",
"shorter please". Requiring an `@mention` on every turn makes the bot feel
like a search box; answering every channel message is the proactive mode
entry 13 deliberately ships silent.

**Decision.** Subscribe to `message.channels`. A reply in a thread is answered
when, and only when, Nazar has already posted in that thread (one
`conversations.replies` read decides it); everything else in the stream is
ACKed and dropped. Mentions keep working anywhere. Before answering, the
thread's earlier turns (capped at 12, 600 characters each, Nazar's own replies
marked as his) are read back, a `gpt-4.1-mini` call rewrites the latest
message as a standalone question and says whether it needs a lookup at all,
and the stance model receives the conversation with instructions to answer the
current turn in its light, not repeat itself, and not reframe what it already
reframed. With no lookup needed ("thanks", "shorter"), no retrieval runs.
`src/query/thread.py`; routing in `app.py`.

**Why.** Being in the thread is the invitation. It keeps entry 13's promise
(the bot never starts a conversation) while making the ones people start with
it feel like conversations. Condensing before retrieval is what makes an
elliptical follow-up retrievable at all; skipping retrieval for
acknowledgements is what keeps the thread cheap.

**Cost.** Every public-channel message in channels the bot is in now hits the
endpoint (dedupe and the ACK path matter more, security open items S1/S2). One
extra cheap model call per follow-up. The thread read needs
`channels:history`, already in the scope set. Enterprise Grid and shared
channels are untested.

**Revisit when.** Threads get long: a running summary instead of the last
twelve turns; and when the bot's own earlier answer should count as a source
the person can be pointed back to.

---

## What did not work

| Idea | Why it was dropped |
|---|---|
| "Wait 20 minutes, everything is ready" | Slack throttling plus extraction throughput make it a promise that cannot be kept. Replaced by 11. |
| Answer any message that looks like a question | Uninstall risk dominates answer quality. Replaced by 13. |
| One file per message | Under 6 alone, most files would be one-line messages. Replaced by 7. |
| LLM-classify every message | Most workspace traffic is not a question; the prefilter in 14 is ~free. |
| Fold attachments into the thread's text | Loses cross-channel dedupe and the `render_version` re-render path. Replaced by 10. |

---

## Open items

**A. Retention and deletion.** Uninstall revokes access (security.md entry 8),
but says nothing about deleting the corpus. Marketplace review will ask, and
GDPR-style deletion requests need an answer.

**B. Per-channel ACL.** Isolation is tenant-level. Anyone who can query the
tenant reaches everything in it. This blocks private channels (5) and is a real
concern even for public ones once a channel is archived or restricted.

**C. `connector_channels` shape.** Cursor semantics, and how channel discovery
reacts to a channel created after install.

**D. Slack rate limits — verify.** Every throughput claim in 11 is *unverified*
and load-bearing for the launch plan. Confirm current tiers, and what Marketplace
approval actually changes, before committing to a backfill design.

**E. Canvases and Posts.** Slack-native documents, not reachable via
`url_private`. Unmodelled.

**F. Vision on images.** Slack images skew to screenshots of errors and
dashboards — high value, high cost at workspace scale. Probably opt-in.

**G. Message deletion and edits.** 8 covers the cost of edits. Nothing yet
specifies how a deleted message is retired from an already-indexed thread.

**H. Bot identity in the graph.** The bot's own answers are messages in threads.
Re-indexing them would let the system cite itself as a source.
