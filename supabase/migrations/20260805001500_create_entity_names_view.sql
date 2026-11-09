-- Entities with their names resolved, for browsing and for any read path that
-- wants a name without joining for it.
--
-- A VIEW rather than a column on `entities`, and that is the whole point: the
-- name is derived from members, so storing it makes a cache that re-clustering
-- silently invalidates. It would also collapse the only distinction
-- `display_name` carries -- NULL means "nobody corrected this", non-NULL means a
-- person did. Fill it with the derived value and a later re-cluster either
-- overwrites a real correction or keeps a stale name, with no way to tell which.
--
-- security_invoker so the tenant walls on entities and chains apply to whoever
-- queries it. Without it the view would run as its owner and read every tenant.
CREATE VIEW public.entity_names WITH (security_invoker = true) AS
SELECT e.tenant_id,
       e.id,
       coalesce(e.display_name,
                public.entity_display_name(e.tenant_id, e.id)) AS name,
       e.display_name IS NOT NULL           AS is_override,
       count(DISTINCT c.id)                 AS chains,
       count(DISTINCT c.episode_id)         AS episodes,
       e.created_at
FROM public.entities e
LEFT JOIN public.chains c
       ON (c.tenant_id, c.entity_id) = (e.tenant_id, e.id)
GROUP BY e.tenant_id, e.id, e.display_name, e.created_at;

-- No GRANT here: app_user does not exist until 20260806183343, and that
-- migration's GRANT ... ON ALL TABLES IN SCHEMA public covers views as well as
-- tables, so the server can read this view once the roles land.

-- entity_display_name() runs once per row here, so this is for browsing and for
-- entity lists, not for a hot loop over the whole table. An entity with no
-- chains yields NULL and chains = 0 -- that is a merged-away or emptied entity,
-- and seeing it is the point.
