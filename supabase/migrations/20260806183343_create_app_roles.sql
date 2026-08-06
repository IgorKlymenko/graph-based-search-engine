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

-- Both inherit the table grants below, so the privilege set stays in one place
-- and the roles differ only in how a tenant is resolved.
GRANT app_user TO app_connector;
GRANT app_user TO app_ingest;

-- Without this nothing can `SET ROLE` into them -- "permission denied to set
-- role" and the whole bundle is unusable.
--
-- postgres carries BYPASSRLS, so the walls do NOT apply until the SET ROLE has
-- happened: the switch is what enforces isolation, not the connection. Any
-- backend path that forgets it runs unwalled. In production give the server its
-- own LOGIN role, granted app_connector or app_ingest and *without* BYPASSRLS,
-- so the floor is a walled role rather than a superuser.
GRANT app_user TO postgres;
GRANT app_connector TO postgres;
GRANT app_ingest TO postgres;

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

-- ── resource ceilings ────────────────────────────────────────────────────────

-- A runaway query otherwise pins a pooled connection indefinitely, and an
-- abandoned transaction holds its locks.
--
-- CAVEAT: ALTER ROLE ... SET is applied at *login*, against the role that
-- authenticated. These roles are NOLOGIN and reached by SET ROLE, so these do
-- NOT take effect today -- verified, statement_timeout reads back as 0 inside a
-- SET ROLE session. They start working the moment the server connects as a
-- dedicated LOGIN role granted one of them, which is the production shape
-- anyway. Until then the ceiling must come from the connection string
-- (`options=-c statement_timeout=30s`) or a SET LOCAL beside the SET LOCAL ROLE.
ALTER ROLE app_user SET statement_timeout = '30s';
ALTER ROLE app_user SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE app_connector SET statement_timeout = '30s';
ALTER ROLE app_connector SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE app_ingest SET statement_timeout = '30s';
ALTER ROLE app_ingest SET idle_in_transaction_session_timeout = '10s';
