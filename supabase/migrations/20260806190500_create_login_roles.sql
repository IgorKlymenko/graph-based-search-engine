-- The roles the server actually authenticates as. Everything above this point
-- is a privilege bundle; these are credentials.
--
-- Two problems close here, and they close together:
--
-- 1. RESOURCE CEILINGS. `ALTER ROLE ... SET` is applied at *login*, against the
--    role that authenticated. app_connector and app_ingest are NOLOGIN and
--    reached by SET ROLE, so ceilings set on them never fire -- verified,
--    statement_timeout reads back as 0 inside a SET ROLE session. Set on a role
--    that logs in, they work. Note that per-role settings are NOT inherited
--    through membership, so they have to be repeated here rather than left on
--    the parent.
--
-- 2. THE BYPASSRLS FLOOR. Until now the only role that could reach the bundle
--    was postgres, which carries BYPASSRLS: a code path that forgot its SET ROLE
--    ran with every wall bypassed, silently and across all tenants. These roles
--    have no BYPASSRLS, so the worst case is no longer "sees everything".
--
-- NOINHERIT is the load-bearing part. Without it a login role granted
-- app_connector would hold those privileges automatically, SET ROLE would be
-- optional, and current_setting('role') would stay 'none' -- which the resolver
-- reads to tell ingest from connector, so the ingest branch would never fire and
-- every upload would silently resolve to NULL. With NOINHERIT the login role can
-- do nothing but connect until it names a role, so:
--
--   * forgetting SET ROLE fails loudly with permission denied, instead of
--     running unwalled (postgres) or running walled-but-misresolved (INHERIT);
--   * the resolver sees the role the worker actually chose.
--
-- No password. A LOGIN role without one cannot authenticate under scram, so
-- these are inert until a password is set out of band -- never in a migration,
-- which lands in git and in the statement log. Set it from your secret store:
--
--     ALTER ROLE app_connector_login PASSWORD '...';
CREATE ROLE app_connector_login LOGIN NOINHERIT IN ROLE app_connector;
CREATE ROLE app_ingest_login    LOGIN NOINHERIT IN ROLE app_ingest;
--Separate from the connector login on purpose: sharing one would let the public
--handler SET ROLE app_connector and reach the corpus anyway, which is the exact
--thing app_ack exists to prevent.
CREATE ROLE app_ack_login       LOGIN NOINHERIT IN ROLE app_ack;

-- A runaway query otherwise pins a pooled connection indefinitely, and an
-- abandoned transaction holds its locks. These are the ones that take effect.
ALTER ROLE app_connector_login SET statement_timeout = '30s';
ALTER ROLE app_connector_login SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE app_ingest_login    SET statement_timeout = '30s';
ALTER ROLE app_ingest_login    SET idle_in_transaction_session_timeout = '10s';
--Tighter: the ACK path does one insert inside a 3-second Slack budget. A query
--there taking longer than five seconds has already failed its real deadline.
ALTER ROLE app_ack_login       SET statement_timeout = '5s';
ALTER ROLE app_ack_login       SET idle_in_transaction_session_timeout = '5s';
ALTER ROLE app_ack             SET statement_timeout = '5s';
ALTER ROLE app_ack             SET idle_in_transaction_session_timeout = '5s';

-- SET ROLE does not re-apply per-role settings, so the ceiling established at
-- login survives the switch into app_connector / app_ingest.

-- One runaway worker must not be able to exhaust the connection pool for the
-- others. Sized well above expected concurrency -- this is a blast-radius cap,
-- not a tuning knob, and hitting it should read as a bug rather than load.
ALTER ROLE app_connector_login CONNECTION LIMIT 30;
ALTER ROLE app_ingest_login    CONNECTION LIMIT 10;
--The ACK path is one insert per request but the request rate is Slack's, not
--ours, so it gets the widest allowance and the shortest timeout.
ALTER ROLE app_ack_login       CONNECTION LIMIT 40;

-- Belt and braces: if any bundle is ever given LOGIN directly, it should not
-- arrive without a ceiling.
ALTER ROLE app_user SET statement_timeout = '30s';
ALTER ROLE app_user SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE app_connector SET statement_timeout = '30s';
ALTER ROLE app_connector SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE app_ingest    SET statement_timeout = '30s';
ALTER ROLE app_ingest    SET idle_in_transaction_session_timeout = '10s';
