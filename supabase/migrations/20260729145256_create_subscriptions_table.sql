CREATE TABLE public.subscriptions (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    sub_status              TEXT NOT NULL DEFAULT 'trial' CHECK (sub_status IN ('active', 'cancelled', 'trial', 'past_due', 'expired')),
    current_period_end      TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON public.subscriptions (tenant_id);
CREATE UNIQUE INDEX one_active_sub_per_tenant
    ON public.subscriptions (tenant_id)
    WHERE sub_status IN ('active','trial');

ALTER TABLE public.subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.subscriptions FORCE ROW LEVEL SECURITY;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION extensions.moddatetime(updated_at);