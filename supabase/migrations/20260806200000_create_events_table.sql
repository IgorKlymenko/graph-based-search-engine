-- Inbound webhook events: dedupe at ACK, idempotency at /work, replay after.
--
-- In `internal` rather than `public` because PostgREST exposes public
-- (config.toml: schemas = ["public", "graphql_public"]) and this table holds raw
-- customer payloads. One misconfigured grant should not turn it into an API.
-- RLS below is then a second line rather than the only one.
CREATE SCHEMA IF NOT EXISTS internal;

-- Keyed like connector_installations, not on slack_event_id: Drive and Notion
-- webhooks have the same retry problem, and a Slack-specific key would mean a
-- second table with the same shape the first time one arrives.
CREATE TABLE internal.events (
    source       TEXT NOT NULL CHECK (source IN ('slack','gdrive','notion')),
    external_id  TEXT NOT NULL, --the workspace. This is what the wall reads.
    event_id     TEXT NOT NULL, --the source's own event id
    status       TEXT NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued','processing','done','failed')),
    --distinguishes "we were slow" from "we failed": Slack's X-Slack-Retry-Num.
    --Without it every duplicate looks alike and a timeout problem is
    --indistinguishable from a crash loop.
    retry_num    INT NOT NULL DEFAULT 0,
    payload      JSONB, --for replay; the largest column, hence the retention rule
    --Metadata, NOT the isolation key. The tenant is not known at ACK -- only the
    --workspace is -- and a nullable tenant_id cannot be walled, because
    --`NULL = anything` is NULL rather than true. Wall on the workspace instead
    --(see the policy below); fill this in at /work if it is useful.
    tenant_id    UUID,
    error        TEXT,
    received_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    PRIMARY KEY (source, external_id, event_id)
);

--This is the only table that grows with traffic rather than with corpus, and
--`payload` makes rows large. Retention decided now, not retrofitted onto a big
--table: sweep by received_at from a privileged job.
CREATE INDEX ON internal.events (received_at);

--finding stuck work without scanning the finished majority
CREATE INDEX ON internal.events (status) WHERE status <> 'done';

ALTER TABLE internal.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE internal.events FORCE ROW LEVEL SECURITY;

-- ── the wall: workspace, not tenant ──────────────────────────────────────────

-- Read from the setting directly rather than through current_tenant_id(): at
-- ACK time no tenant has been resolved, and the workspace is precisely what the
-- verified signature established. Unset settings make both sides NULL, so the
-- comparison is NULL rather than true and the wall denies -- fail-closed, same
-- as every other table.
CREATE POLICY events_workspace_wall ON internal.events
    AS RESTRICTIVE FOR ALL TO app_ack, app_user
    USING (source = nullif(current_setting('app.connector', true), '')
       AND external_id = nullif(current_setting('app.workspace_id', true), ''))
    WITH CHECK (source = nullif(current_setting('app.connector', true), '')
            AND external_id = nullif(current_setting('app.workspace_id', true), ''));

-- ── doors, split by which half of the flow is asking ─────────────────────────

--The ACK path only ever inserts. It never needs to read one back: the whole
--dedupe is `INSERT ... ON CONFLICT DO NOTHING` and a rowcount of zero.
CREATE POLICY events_ack_insert ON internal.events
    FOR INSERT TO app_ack WITH CHECK (true);

--The worker claims and completes. No INSERT: an event that was never ACKed
--should not appear from the work side.
CREATE POLICY events_worker_read ON internal.events
    FOR SELECT TO app_user USING (true);
CREATE POLICY events_worker_claim ON internal.events
    FOR UPDATE TO app_user USING (true) WITH CHECK (true);

-- ── grants ───────────────────────────────────────────────────────────────────

GRANT USAGE ON SCHEMA internal TO app_ack, app_user;
GRANT INSERT ON internal.events TO app_ack;
GRANT SELECT, UPDATE ON internal.events TO app_user;
--Nobody deletes through the app roles; retention is a privileged sweep.

-- The claim is one atomic statement, which is what makes a duplicate Cloud Tasks
-- delivery safe without a lock:
--
--     UPDATE internal.events SET status = 'processing', started_at = now()
--      WHERE source = $1 AND external_id = $2 AND event_id = $3
--        AND status IN ('queued','failed')
--     RETURNING payload;
--
-- Zero rows means someone else holds it, or it is done. Stop.
