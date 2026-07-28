CREATE TABLE public.tenants (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL CHECK (length(trim(name)) > 0),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tenants FORCE ROW LEVEL SECURITY;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.tenants
    FOR EACH ROW
    EXECUTE FUNCTION extensions.moddatetime(updated_at);