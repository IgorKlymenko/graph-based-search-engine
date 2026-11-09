# Slack

Two halves: the **ingress** that Slack posts to (`server/public/src/app.py`) and
the **connector** that pulls history and renders it into documents
(`old_server/entity_linking/slack.py`). Design: `ADRs/slack/connector.md`;
request-path decisions: `ADRs/supabase/tables.md` entries 6–10, 14–17.

## Ingress: `POST /slack/event`

Order inside the handler is the security design; do not reorder:

1. `verify_signature(headers, raw)` — HMAC-SHA256 over `v0:{ts}:{raw body}` with `SLACK_SIGNING_SECRET`, constant-time compare, timestamp older than 5 minutes rejected (unless `VERIFY_TIMESTAMP=false`, dev only — ngrok's Replay button resends the original timestamp). Runs on **raw bytes before JSON parsing**; re-serialising changes the signature.
2. `url_verification` → return the `challenge` as plain text. Must return a body, so it branches before the ACK path.
3. Dedupe on `event_id` — today an in-memory set that resets on reload; the production mechanism is `INSERT INTO internal.events … ON CONFLICT DO NOTHING` as `app_ack` (see `CLAUDE.md` § Webhook handling).
4. Drop `bot_id` and `subtype == bot_message` so the bot never answers itself.
5. `app_mention` anywhere, and `message` events that are replies in a thread (`thread_ts` set, `channel_type == channel`, no subtype) → `BackgroundTasks` (stand-in for a Cloud Tasks enqueue) → return `{}`. A thread reply is answered only if Nazar already posted in that thread (`handle_thread_reply`, ADR slack 17); other channel traffic is ACKed and dropped.

Replies are Block Kit: answer section, a divider, then a muted context line `*Sources* · <url|title> (code) · title (note) · +N more` (`answer._sources_line`); code sources show the path's last three segments and link to github.com.

`handle_event(event, team_id, turns)` then answers in the thread. Inside a thread it first reads the earlier turns (`thread.thread_context`, capped) and condenses the message into a standalone question with a `needs_lookup` flag (`thread.condense`, gpt-4.1-mini); `answer_question` gets the conversation and skips retrieval when no lookup is needed. Then: `question_from` strips the mention and resolves Slack markup; `retrieve.answer_question` runs as `app_connector` with `app.workspace_id = team_id`, so the database resolves the tenant and an unknown or uninstalled workspace finds nothing; `answer.render_slack_blocks` posts the stance as Block Kit: a reframe is a two-line *Heads-up* / *Better question* section above the answer section, a clarification is a bulleted question, and sources are a muted context line of readable note titles (a note's stem, linked when `files.url` is known). Markdown the model emits is converted to mrkdwn, sections are clipped to Slack's 3,000-character block limit, link unfurling is off, the prompt's FORMAT rules keep answers under about 120 words, and `answer._without_preface` removes any sentence that still restates the preface (the model did so about half the time). Nothing matched or any failure → a neutral fallback, with the exception logged by workspace id and never the message text. Needs `CONNECTOR_DATABASE_URL` (an `app_connector_login` DSN) and a `connector_installations` row for the team; `ANSWER_MODEL` is optional. **Verified live 2026-09-18**: with the vault loaded as tenant `dddddddd-…` and `T0BBWPQ8BEC` installed, a signed `app_mention` posted to the local ingress produced a threaded reframe in the channel within seconds ("Wrong question, per the notes: Neo4j Aura's linear per-tenant pricing is considered unscalable…"), with ngrok on the reserved domain answering `/health` publicly. Each answer logs `workspace=… channel=… stance=… ts=…` and never the text.

`GET /health` is unauthenticated and must stay cheap; never hang model or graph loading off it.

Env (`server/public/.env.example`): `SLACK_SIGNING_SECRET`, `SLACK_BOT_TOKEN` (`xoxb-`), `VERIFY_TIMESTAMP`, `OPENAI_API_KEY`, `DATABASE_URL` (ingest login), `CONNECTOR_DATABASE_URL` (connector login; the worker's DSN), `ANSWER_MODEL`.

Tests: `server/tests/public/auth/` — valid, tampered body, missing signature, bad timestamp, stale timestamp, missing timestamp, health; `server/tests/public/query/test_slack_handler.py` — a mention answered in thread as its workspace, fallback on nothing found, a failing retrieval still replies and logs no text, an empty mention, markup resolution. `slack_sign` in `src/security/slack.py` is the shared signer for tests and, in future, outbound verification.

## Event payload shape (`slack/event_request_example.json`)

Top level: `token, team_id, api_app_id, event, type, event_id, event_time, authorizations, is_ext_shared_channel, event_context`. `event`: `type, user, ts, client_msg_id, text, team, blocks, channel, event_ts`. Threads add `thread_ts`. The workspace is `team_id`; the sender is `event.user`; the dedupe key is `event_id`; retries carry the `X-Slack-Retry-Num` header (stored as `events.retry_num`).

## Manifest (`slack/manifest.dev.yaml`) vs the ADR

| | Manifest today | ADR slack 4 says |
|---|---|---|
| Scopes | `app_mentions:read`, `channels:history`, `channels:read`, `chat:write`, `files:read`, `users:read` (aligned 2026-09-17) | the same list; **never** `im:history` / `mpim:history` |
| Events | `app_mention`, `app_home_opened` | `app_mention` + slash command at launch; proactive per channel, silent first |
| Other | `org_deploy_enabled: true`, `token_rotation_enabled: false` | Enterprise Grid tenancy is an open question (`ADRs/brainstorming.md`) |
| Identity | name **Loupy**, bot `@loupy`, always online, icon `slack/assets/loupy-icon.png` (the company mark on white), background `#1D1D1F`; set in the dashboard on 2026-09-19 and mirrored in the manifest | the dev app was "Nazar" until then; the `@` handle only changes on reinstall, which mints a new bot token |

`chat:write.public` was dropped with the alignment, so the bot must be invited to a channel before it can reply there. Reinstall after any scope change.

## Connector: history → documents

`python -m entity_linking.slack --tenant <uuid> --store <path> [--days 30] [--channels a b] [--pause 1.2]`. Token from `SLACK_BOT_TOKEN`; preflights with `auth.test` so a bad token fails with a name attached.

| Rule | Where | Why |
|---|---|---|
| A thread with replies → `thread/{channel_id}/{thread_ts}`; loose messages → `day/{channel_id}/{YYYY-MM-DD}` | `documents()` | ADR slack 6, 7 |
| One message = one paragraph `[HH:MM] Name: text`, blank-line separated | `render_message` | appends re-extract **one** episode (ADR slack 8, measured) |
| `<@U…>` → `@name`, `<#C…|eng>` → `#eng`, `<url|text>` → `text`, `<!here>` → `@here`, HTML entities unescaped — **before hashing** | `resolve_text` | contexts must be findable by exact search in their episode (ADR slack 9) |
| Drop `SKIP_SUBTYPES` (joins, topic changes, bot_message, message_changed/deleted, …) and any `bot_id`/`app_id` | `is_content` | otherwise every channel grows a "has joined the channel" entity |
| Public channels the bot is a member of, not archived, never `is_im`/`is_mpim` | `Slack.channels` | second line behind the scope decision |
| 1.2 s pause between calls, honour `Retry-After` on 429, page size 200 | `Slack` | non-Marketplace apps are throttled hard (unverified, ADR open item D) |
| Unknown user id renders as `@U0123`, not dropped | `resolve_text` | a mention is an edge worth counting |

Tests: `old_server/tests/test_slack_render.py` (18).

Known gaps, all recorded in the ADR: day-file promotion when a message later gets a reply (silent duplication); attachments as their own files (`url_private_download` needs the bearer token or you store an HTML login page); canvases; edits shift downstream episodes; deleted messages are not retired; no `connector_channels` cursor so an interrupted backfill restarts; no OAuth callback, so install auto-provisioning (ADR slack 1) does not exist.

## Dev loop

```
supabase start                                  # if the worker will write
cd server/public && uv run uvicorn src.app:app --port 8000 --reload
ngrok http 8000                                 # then: curl -s http://127.0.0.1:4040/api/tunnels
# put the https URL into slack/manifest.dev.yaml request_url + redirect_urls, reinstall the app
# or push the manifest from here (needs SLACK_CONFIG_TOKEN + SLACK_APP_ID in .env):
uv run python scripts/slack_manifest_push.py   # --check only validates
# a new event whose scope is already granted takes effect without reinstall
```

Full commands in [runbook.md](runbook.md).
