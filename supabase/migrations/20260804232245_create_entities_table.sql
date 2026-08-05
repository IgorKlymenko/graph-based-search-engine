CREATE TABLE public.entities (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    canonical_name  TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
    --no merged_into tombstone on purpose. Nothing outside the graph holds an
    --entity id: the PPR cache is regenerated on every graph update, and edges
    --are derived from chain co-occurrence rather than stored per entity. So a
    --merge is just chains changing entity_id, and the emptied entity is deleted
    --rather than reokaymdirected. A split mints a fresh id for the new half.
    --Reinstate this only if something starts holding ids ACROSS syncs -- a
    --cached answer, a saved link -- because then a deleted id is unresolvable
    --and there is no way to discover where it went.
);

ALTER TABLE public.entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.entities FORCE ROW LEVEL SECURITY;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.entities
    FOR EACH ROW
    EXECUTE FUNCTION extensions.moddatetime(updated_at);