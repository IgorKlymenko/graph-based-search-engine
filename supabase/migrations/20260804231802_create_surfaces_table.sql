CREATE TABLE public.surfaces (
    tenant_id   UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    surface     TEXT NOT NULL CHECK (length(trim(surface)) > 0),
    embedding   extensions.VECTOR(1536), --null until embedded; backfilled in batches
    embed_model TEXT NOT NULL DEFAULT 'text-embedding-3-large@1536',
    --no n_entities / n_episodes here on purpose: both were measured negative as
    --seeding weights and both are derivable from chain_surfaces JOIN chains,
    --which the seeding query already does. See DATAFLOW.md "Query".
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, surface)
);

CREATE INDEX ON public.surfaces
    USING hnsw (embedding extensions.vector_cosine_ops); --top-k seeding

ALTER TABLE public.surfaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.surfaces FORCE ROW LEVEL SECURITY;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.surfaces
    FOR EACH ROW
    EXECUTE FUNCTION extensions.moddatetime(updated_at);
