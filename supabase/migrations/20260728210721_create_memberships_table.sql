CREATE TABLE public.memberships (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid NOT NULL REFERENCES public.users(id)   ON DELETE CASCADE,
    tenant_id  uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    role       text NOT NULL CHECK (role IN ('owner','admin','member')),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, tenant_id)
);

CREATE INDEX ON public.memberships (tenant_id);

ALTER TABLE public.memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.memberships FORCE ROW LEVEL SECURITY;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.memberships
    FOR EACH ROW
    EXECUTE FUNCTION extensions.moddatetime(updated_at);