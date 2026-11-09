SELECT e.id,
       coalesce(e.display_name,
                public.entity_display_name(e.tenant_id, e.id)) AS name,
       e.display_name AS override,
       count(DISTINCT c.id) AS chains
FROM entities e
LEFT JOIN chains c ON (c.tenant_id, c.entity_id) = (e.tenant_id, e.id)
WHERE e.tenant_id = 'dddddddd-0000-4000-8000-00000000000d'
GROUP BY e.id, e.tenant_id, e.display_name
ORDER BY chains DESC;