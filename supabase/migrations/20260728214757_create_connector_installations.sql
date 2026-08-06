-- One row per customer install, whatever the source. This is the table the
-- tenant wall consults, and the reason adding Notion or Drive later needs no
-- DDL, no role and no policy -- only a row.
--
-- Replaces a Slack-specific `slack_teams`, which held nothing this does not:
-- team id, tenant, token_ref, installed/uninstalled. Keying on (source,
-- external_id) generalises it at no cost, and one lookup in the resolver beats
-- a CASE that grows a branch per connector on the hot path of every policy.
--
-- The trust model is identical for every source here: an OAuth install yields an
-- external id the server can verify, and the database maps it to a tenant. That
-- only holds if the id really is verified on the way in -- a webhook whose
-- workspace id is unauthenticated is a bare tenant claim wearing a connector's
-- clothes, and nothing downstream can tell the difference.
CREATE TABLE public.connector_installations (
    source         TEXT NOT NULL CHECK (source IN ('slack','gdrive','notion')),
    external_id    TEXT NOT NULL, --team id, workspace id, domain: the source's own
    tenant_id      UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    token_ref      TEXT NOT NULL, --pointer into the secret store, never the token
    installed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    uninstalled_at TIMESTAMPTZ, --set on uninstall; access stops the same moment
    PRIMARY KEY (source, external_id),
    UNIQUE (source, external_id, tenant_id) --composite FK target for slack_users
);

CREATE INDEX ON public.connector_installations (tenant_id);

ALTER TABLE public.connector_installations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.connector_installations FORCE ROW LEVEL SECURITY;

-- No policy, ever. Workers reach this only through the SECURITY DEFINER
-- resolvers, which return one tenant or one token_ref for the workspace the
-- caller has already proven. A readable mapping of every workspace to its
-- tenant is the customer list; the grant is revoked in the roles migration.
