-- The roles that touch embeddings need the `vector` type, which lives in the
-- `extensions` schema.
--
-- Without USAGE on that schema, a type inside it does not merely fail a
-- permission check -- it reports as *nonexistent*:
--
--     psycopg.errors.UndefinedObject: type "vector" does not exist
--
-- which reads like a missing extension rather than a missing grant, and sends
-- you looking in entirely the wrong place. `search_path` already lists
-- `extensions`; a schema you cannot use is invisible regardless of the path.
--
-- This stayed hidden until something first wrote an embedding. The extraction
-- pipeline sets `embed_model` but leaves `embedding` NULL, so every write path
-- exercised until now avoided the type completely.

-- Granted to the two roles that actually handle vectors, NOT to `app_user`.
-- `app_user` is the shared privilege bundle, so granting there would also admit
-- `app_ack` -- the role on the public webhook, the most exposed surface we have
-- and the one with no business reading anything. Keeping it out costs nothing:
-- it inserts one row into `internal.events` and never sees an embedding.
GRANT USAGE ON SCHEMA extensions TO app_ingest;
GRANT USAGE ON SCHEMA extensions TO app_connector;

-- KNOWN RESIDUAL EXPOSURE, deliberately not "fixed" here.
--
-- `extensions` also holds `pg_stat_statements`, which Supabase ships with SELECT
-- granted to PUBLIC (`=r/supabase_admin` in relacl). PUBLIC includes the two
-- roles above, so this grant makes that view readable to them.
--
-- It cannot be closed from a migration. The grant was made *by* `supabase_admin`
-- and only that role can revoke it; `postgres` cannot `SET ROLE supabase_admin`,
-- and a REVOKE issued as `postgres` reports success while changing nothing --
-- which is worse than leaving it out, because it looks like a control.
--
-- What is actually exposed: pg_stat_statements normalises literals, so entries
-- read `... WHERE tenant_id = $1`, not the id. No tenant *data* leaks. What does
-- leak is the shape of activity -- table names, query forms, call counts and
-- timings, database-wide. Judged acceptable against needing `vector` at all,
-- and narrowed by keeping `app_ack` out.
--
-- To close it properly: revoke as `supabase_admin` via the Supabase dashboard or
-- support, or relocate pgvector to a schema of our own. Tracked in
-- `ADRs/supabase/security.md`.

-- No blanket grant on functions in `extensions`. USAGE on the schema gives the
-- vector type and its operators, which is the whole requirement; `pg_net` --
-- outbound HTTP from inside the database -- stays unreachable by default.
