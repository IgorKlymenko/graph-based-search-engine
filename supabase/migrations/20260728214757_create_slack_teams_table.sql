CREATE TABLE public.slack_teams (
    id              TEXT PRIMARY KEY, 
    tenant_id       UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    token_ref       text NOT NULL,
    installed_at    timestamptz NOT NULL DEFAULT now(),
    uninstalled_at  timestamptz DEFAULT NULL
);

CREATE INDEX ON public.slack_teams (tenant_id);

ALTER TABLE public.slack_teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.slack_teams FORCE ROW LEVEL SECURITY;

--if reinstalled: clean the uninstalled_at and udpate the installed_at