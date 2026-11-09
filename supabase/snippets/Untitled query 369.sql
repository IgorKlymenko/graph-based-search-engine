BEGIN;
SET LOCAL ROLE app_ingest;
SET LOCAL app.tenant_id = 'aaaaaaaa-0000-4000-8000-00000000000a';

SELECT public.entity_display_name(e.tenant_id, e.id) AS entity,
       count(DISTINCT c.id) AS chains,
       string_agg(DISTINCT cs.surface, ', ') AS surfaces
  FROM entities e
  JOIN chains c  ON (c.tenant_id, c.entity_id) = (e.tenant_id, e.id)
  JOIN chain_surfaces cs ON (cs.tenant_id, cs.chain_id) = (c.tenant_id, c.id)
 GROUP BY e.tenant_id, e.id
 ORDER BY chains DESC;
COMMIT;
