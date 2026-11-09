SELECT chain_id, count(*)
  FROM chain_surfaces
  GROUP BY chain_id
  ORDER BY count(*) DESC;