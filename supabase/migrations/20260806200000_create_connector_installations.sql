-- One row per customer install, whatever the source. This is the table the
-- tenant wall consults, and the reason adding Notion or Drive needs no DDL, no
-- role and no policy change -- only a row.
--
-- `slack_teams` stays as the Slack-specific detail (slack_users FKs to it, bot
-- scopes live there). This carries only what resolution needs: which workspace,
-- whose tenant, still installed.
--
-- The trust model is the same for every source here: an OAuth install gives an
-- external id the server can verify, and the database maps it to a tenant. That
-- only holds if the id really is verified on the way in -- a webhook whose
-- workspace id is unauthenticated is a bare claim wearing a connector's
-- clothes, and nothing below can tell the difference.
CREATE TABLE public.connector_installations (
    source         TEXT NOT NULL CHECK (source IN ('slack','gdrive','notion')),
    external_id    TEXT NOT NULL, --team id, workspace id, domain: the source's own
    tenant_id      UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    token_ref      TEXT NOT NULL, --pointer into the secret store, never the token
    installed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    uninstalled_at TIMESTAMPTZ, --set on uninstall; access stops the same moment
    PRIMARY KEY (source, external_id)
);

CREATE INDEX ON public.connector_installations (tenant_id);

ALTER TABLE public.connector_installations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.connector_installations FORCE ROW LEVEL SECURITY;

-- ALTER DEFAULT PRIVILEGES (see the app_user migration) grants every new table
-- in this schema to app_user automatically, and it fired for this one. RLS then
-- returns zero rows rather than raising, so the hole reads as "working". Take
-- the grant back explicitly -- a table whose rows map every workspace to its
-- tenant is the customer list.
REVOKE ALL ON public.connector_installations FROM app_user;

-- No policy and no grant, deliberately. The workers reach it only through the
-- SECURITY DEFINER resolvers, which return one tenant or one token_ref for the
-- declared workspace. A direct SELECT grant here would let any worker enumerate
-- every workspace and its tenant mapping -- the whole customer list.

-- Backfill from the existing Slack installs so the resolver has one source of
-- truth from the first run.
INSERT INTO public.connector_installations
       (source, external_id, tenant_id, token_ref, installed_at, uninstalled_at)
SELECT 'slack', id, tenant_id, token_ref, installed_at, uninstalled_at
FROM public.slack_teams
ON CONFLICT (source, external_id) DO NOTHING;
