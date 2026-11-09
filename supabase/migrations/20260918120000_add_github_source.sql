-- GitHub joins the sources. Same trust model as Slack (ADRs/supabase/security.md
-- entry 4): a GitHub App installation yields an installation id the server can
-- verify, and connector_installations maps it to a tenant. A row, not a role.
--
-- The CHECK lists are the one place a source name is enumerated in the schema.
-- They are recreated rather than dropped: an open list would let a typo in
-- application code mint a source nothing renders, and files.source feeds
-- render_version, which is per source.
--
-- Constraint names are the ones Postgres generated for the inline CHECKs
-- (<table>_<column>_check). Until this is applied, a GitHub sync fails at the
-- files upsert with a check_violation, which is the loud failure we want.
ALTER TABLE public.connector_installations
    DROP CONSTRAINT connector_installations_source_check,
    ADD CONSTRAINT connector_installations_source_check
        CHECK (source IN ('slack','gdrive','notion','github'));

ALTER TABLE public.files
    DROP CONSTRAINT files_source_check,
    ADD CONSTRAINT files_source_check
        CHECK (source IN ('slack','gdrive','notion','upload','github'));

ALTER TABLE internal.events
    DROP CONSTRAINT events_source_check,
    ADD CONSTRAINT events_source_check
        CHECK (source IN ('slack','gdrive','notion','github'));
