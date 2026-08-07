CREATE TABLE public.entities (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    --A human override, and ONLY that. The entity's name is derived from its
    --members -- see public.entity_display_name() -- exactly as a chain's head is
    --a flag on chain_surfaces rather than a string copied onto chains. A stored
    --name would be a denormalised copy that re-clustering silently overwrites,
    --and a nullable *cache* would be worse still: you could not tell a stale
    --value from a deliberate correction. Null here means "nobody has corrected
    --this", which is a question the derivation cannot answer for itself.
    display_name    TEXT CHECK (display_name IS NULL
                               OR length(trim(display_name)) > 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
    --no merged_into tombstone on purpose. Nothing outside the graph holds an
    --entity id: the PPR cache is regenerated on every graph update, and edges
    --are derived from chain co-occurrence rather than stored per entity. So a
    --merge is just chains changing entity_id, and the emptied entity is deleted
    --rather than redirected. A split mints a fresh id for the new half.
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