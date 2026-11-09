Blocks writing any code

- Slack rate limits — verify current tiers. Every throughput number in the spec is unverified and the launch plan rests on it. Also: what exactly does Marketplace approval change, and can you get approved before having a working backfill?
- What a "workspace" means for pricing. One tenant per Slack team is assumed throughout. Enterprise Grid has multiple workspaces under one org — one tenant or several?
- Day-file promotion rule. A message that gets a reply later — does it leave the day file? Left open, silently duplicates content.
- Where thread conversation state lives for clarification round-trips: internal.events, or its own table.

Product / UX — mostly easy, but they change scope

- Onboarding: does the installer pick channels, or is it all-public-by-default with opt-out?
- What the app posts on install — DM to installer, channel message, nothing?
- What "ready" means to the user, given progressive indexing.
- Answer format: threaded reply, ephemeral, with citations as links?
- Does the bot answer in the channel where asked, or always in thread?
- Slash command surface: /ask, or message shortcut, or both?
- What happens when it has no answer — silence, or "I don't know"?
- Free trial length and what happens at expiry.

Data model

- Enterprise Grid → external_id shape.
- Do bot answers get indexed? (self-citation risk)
- Channel renames and archives — what happens to indexed content?
- Deleted messages — retire from an indexed thread how?
- files.title for a day file — what's a useful title?
- Do you keep the raw Slack payload anywhere, or only rendered text?
- Reactions, pins, bookmarks — signal worth capturing, or noise?

Cost & scale — the ones I'd model before committing

- Cost per workspace at install (backfill) vs steady state. This determines pricing and I don't think anyone knows it yet.
- What happens on a 10k-message-per-day workspace — extraction queue depth, worker count.
- Vision on images: opt-in, off, or sampled?
- Do you cap indexing per tenant, and what's the behaviour at the cap?
- Re-clustering cadence — entity ids churn, and Slack adds data continuously.

Security & compliance — Marketplace review will ask

- Retention and deletion on uninstall. Open item A; likely a review blocker.
- Per-channel ACL vs tenant-wide. Currently anyone querying the tenant reaches everything. Blocks private channels and is arguably already a problem for public ones.
- Data residency / subprocessor disclosure (you're sending workspace content to OpenAI).
- Admin controls: can a workspace admin see what's indexed, or purge a channel?
- What you log, and whether message text appears in logs.

Ops

- Cloud Tasks OIDC on the worker endpoint — carried over from security.md, still open.
- Slack signature verification + timestamp replay window.
- Backfill failure handling: one channel 403s, does the whole install fail?
- Token rotation / revocation detection (invalid_auth mid-backfill).
- The token_ref secret store — which one, and who can read it.

Things I'd flag as easy but worth 5 minutes each

- Bot display name, avatar, description copy.
- Which scopes appear on the consent screen (users read this).
- Whether groups:history appears greyed-out or absent in v1.
- Error message when someone @mentions in a non-indexed channel.

If you want a starting order: the four blockers, then retention + ACL (because they can force schema changes), then cost modelling, then the product surface. The rate-limit verification is the single highest-value hour — it's the one that could invalidate the backfill design entirely.



Last login: Mon Aug 10 14:54:29 on ttys007
cd /Users/igor/Desktop/knowledgeGraphs/core/old_server
set -a && . ../.env && set +a
(base) igor@MacBook-Air-Igor core % cd /Users/igor/Desktop/knowledgeGraphs/core/old_server
(base) igor@MacBook-Air-Igor old_server % set -a && . ../.env && set +a
.: no such file or directory: ../.env
(base) igor@MacBook-Air-Igor old_server % uv run python -m entity_linking.pipeline \
  --store /tmp/vault.json --notes-dir /tmp/vault_notes \
  --tenant-id ffffffff-0000-4000-8000-00000000000f \
  --dsn "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
[1/4] episodes: 144 documents, 0 changed -> +0 episodes, 0 retired
[1] candidates: 2 episodes to extract (253 in store, 2 pending total)
  1/2 episodes extracted, 7073 candidates in store
  WARNING: 1 episodes failed and stay pending — they will be retried on the next run
[2/4] chains: 1 episodes extracted, 7073 candidates in store
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
<All keys matched successfully>
[transformers] Detected the usage of `get_extended_attention_mask`: This function is deprecated and will be removed in v5.12.0. Please use the new API in `transformers.masking_utils`
[3/4] encode: 5772 chains, 4384 distinct names, 5774 context vectors (305 non-identifying surfaces dropped, 2 chains left with none)
[4/4] link: 220013 pairs scored, 10274 reviewed, 6888 merged -> 3439 entities
  ambiguous mentions: 138 placed by the graph, 40 left unresolved (no node created)
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/igor/Desktop/knowledgeGraphs/core/old_server/entity_linking/pipeline.py", line 448, in <module>
  File "/Users/igor/Desktop/knowledgeGraphs/core/old_server/entity_linking/pipeline.py", line 422, in main
  File "/Users/igor/Desktop/knowledgeGraphs/core/old_server/entity_linking/pipeline.py", line 422, in <genexpr>
KeyError: 2012
(base) igor@MacBook-Air-Igor old_server % 