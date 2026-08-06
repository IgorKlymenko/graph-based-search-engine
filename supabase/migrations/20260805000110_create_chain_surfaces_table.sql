CREATE TABLE public.chain_surfaces (
    tenant_id  UUID NOT NULL,
    chain_id   UUID NOT NULL,
    surface    TEXT NOT NULL,
    is_head    BOOLEAN NOT NULL DEFAULT false,
    --a surface said five times in a chain is more central than one said once.
    --Unlike the other counts this is NOT recoverable later: with no per-mention
    --rows and no offsets, re-deriving it means re-extracting the episode.
    n_mentions INT NOT NULL DEFAULT 1,
    FOREIGN KEY (tenant_id, chain_id)
        REFERENCES public.chains (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, surface)
        REFERENCES public.surfaces (tenant_id, surface),
    PRIMARY KEY (tenant_id, chain_id, surface)
);

--both halves of the evidence lookup, and the entry point from the surface search
CREATE INDEX ON public.chain_surfaces (tenant_id, surface);

--exactly one head per chain
CREATE UNIQUE INDEX ON public.chain_surfaces (tenant_id, chain_id) WHERE is_head;

ALTER TABLE public.chain_surfaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chain_surfaces FORCE ROW LEVEL SECURITY;