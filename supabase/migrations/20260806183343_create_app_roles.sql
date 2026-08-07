-- Every role, grant and revoke the server uses, in one place.
--
-- Split by *how the tenant is established* -- not by which connector the data
-- came from. Notion and Drive are the same trust model as Slack: an OAuth
-- install gives a verifiable external id and the database derives the tenant.
-- Only uploads and backfills have nothing to derive from.
--
--   app_user       the privilege bundle. Not meant to be assumed directly.
--   app_connector  tenant DERIVED from a verified installation. Cannot claim.
--   app_ingest     tenant CLAIMED by the server. No workspace exists.
--
-- The split is structural rather than advisory. With one role holding both
-- paths, a connector worker that set app.tenant_id instead of app.workspace_id
-- would silently fall back to the weaker model -- no error, no signal, just a
-- claim where a derivation was intended. Separated, that worker cannot make the
-- mistake: its resolver never reads app.tenant_id at all.
CREATE ROLE app_user NOLOGIN;
CREATE ROLE app_connector NOLOGIN;
CREATE ROLE app_ingest NOLOGIN;

-- The ACK path, and deliberately NOT a member of app_user.
--
-- It may be the same server today, which makes this look like ceremony. The
-- boundary it draws is not process, it is exposure: the ACK endpoint is
-- publicly reachable because Slack posts to it, while the worker endpoint is
-- (or will be) OIDC-gated. A logic bug or injection in the public handler
-- running as app_connector could read that workspace's entire corpus; running
-- as app_ack it can insert an event row and nothing else.
--
-- Costs one extra `SET LOCAL ROLE app_ack` on the ACK path. If that ever reads
-- as friction, collapsing it is a two-line change: point the events policies at
-- app_user and drop these two roles. The table is the durable decision; the
-- role is not.
CREATE ROLE app_ack NOLOGIN;

-- Both inherit the table grants below, so the privilege set stays in one place
-- and the roles differ only in how a tenant is resolved.
GRANT app_user TO app_connector;
GRANT app_user TO app_ingest;

-- Without this nothing can `SET ROLE` into them -- "permission denied to set
-- role" and the whole bundle is unusable.
--
-- postgres carries BYPASSRLS, so from *that* role the walls do not apply until
-- SET ROLE has happened -- a path that forgets it runs unwalled. This membership
-- is for migrations and tests only. The server authenticates as one of the
-- NOINHERIT login roles in the next migration, where forgetting SET ROLE fails
-- loudly instead.
GRANT app_user TO postgres;
GRANT app_connector TO postgres;
GRANT app_ingest TO postgres;
GRANT app_ack TO postgres;

GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
-- Only covers tables created by the role that runs this (postgres, since
-- migrations run as postgres). A table created by any other role gets no grant.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;

-- ── take back what ON ALL TABLES over-granted ────────────────────────────────

-- Identity resolution runs before a tenant is known, so these carry no tenant
-- column and can have no wall. They are safe today only because RLS is forced
-- and no policy names app_user -- one accidental policy from an unwalled door.
-- A grant that relies on the absence of a policy is not a decision, it is luck.
REVOKE ALL ON public.users, public.slack_users FROM app_user;

-- The workspace -> tenant mapping is the customer list. Reachable only through
-- the SECURITY DEFINER resolvers, which return one row for the workspace the
-- caller has proven. RLS would return zero rows rather than raising, so the hole
-- would read as "working" -- revoke instead, and let it fail loudly.
REVOKE ALL ON public.connector_installations FROM app_user;

-- Tenancy and billing are read-only to the server; writes go through the
-- privileged path. Enforced by revoking the privilege, not by omitting a
-- permissive write policy: RLS *filters*, it does not raise. With the grant left
-- in place `DELETE FROM tenants` succeeds and silently removes zero rows, and
-- the day someone adds a FOR ALL policy it starts removing real ones.
REVOKE INSERT, UPDATE, DELETE
    ON public.tenants, public.memberships, public.subscriptions FROM app_user;

-- ── function execution ───────────────────────────────────────────────────────

GRANT EXECUTE ON FUNCTION public.current_tenant_id() TO app_user;
-- Only the role that proves a workspace may fetch its credential.
GRANT EXECUTE ON FUNCTION public.current_token_ref() TO app_connector;

-- Resource ceilings live with the LOGIN roles, in the next migration:
-- ALTER ROLE ... SET applies at login, and nothing logs in as these.
