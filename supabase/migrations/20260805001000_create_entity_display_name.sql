-- An entity's name, derived from its members rather than stored on it.
--
-- This is `spans.pick_head` lifted one level, and with better information than
-- any single chain has: how many chains chose that head. If eleven episodes
-- call her "Sarah Chen" and one calls her "S. Chen", frequency settles it.
-- Length and lexical order only break ties, so the answer is deterministic --
-- the same members always produce the same name, which matters because
-- re-clustering changes membership constantly.
--
-- Only `is_head` surfaces are considered. A chain's non-head surfaces are the
-- other ways it happened to be written; promoting one of those to the entity's
-- name would surface an alias over the name the extractor actually chose.
--
-- NOT security definer, deliberately: it reads walled tables as the caller, so
-- asking for an entity outside your tenant returns NULL rather than a name.
CREATE FUNCTION public.entity_display_name(p_tenant uuid, p_entity uuid)
RETURNS text
LANGUAGE sql STABLE
SET search_path = '' AS $$
    SELECT coalesce(
        --a human correction wins outright, and survives re-clustering because
        --nothing recomputes into this column
        (SELECT e.display_name
           FROM public.entities e
          WHERE e.tenant_id = p_tenant AND e.id = p_entity),
        (SELECT cs.surface
           FROM public.chains c
           JOIN public.chain_surfaces cs
             ON (cs.tenant_id, cs.chain_id) = (c.tenant_id, c.id)
          WHERE c.tenant_id = p_tenant
            AND c.entity_id = p_entity
            AND cs.is_head
          GROUP BY cs.surface
          ORDER BY count(*) DESC, length(cs.surface) DESC, cs.surface
          LIMIT 1)
    );
$$;

-- Cost: a join per name displayed. It is the same join seeding and passage
-- lookup already perform, and entity lists are short. If it ever shows up in a
-- profile the fix is a materialised stats view alongside the surface bridge --
-- not a column on `entities`, which is what this replaced.
