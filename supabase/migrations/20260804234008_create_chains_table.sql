CREATE TABLE public.chains (
    id                      UUID NOT NULL,
    tenant_id               UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    episode_id              UUID NOT NULL,
    entity_id               UUID, --null before stage 4, and stays null for a contested mention we refuse to make a node
    content_key             TEXT NOT NULL, --sha1 of the sorted surface set; lets re-extraction keep entity_id when the chain is unchangeds
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, episode_id) 
        REFERENCES public.episodes (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, entity_id) 
        REFERENCES public.entities (tenant_id, id) ON DELETE RESTRICT,
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX ON public.chains (episode_id); --"all chains of this episode" during sync
CREATE INDEX ON public.chains (tenant_id, entity_id); --entity -> chains -> episodes
CREATE UNIQUE INDEX ON public.chains (tenant_id, episode_id, content_key); --same chain twice in one episode is a bug

ALTER TABLE public.chains ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chains FORCE ROW LEVEL SECURITY;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.chains
    FOR EACH ROW
    EXECUTE FUNCTION extensions.moddatetime(updated_at);
