CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA extensions;

--Seeding fuses a vector search with a lexical one. The HNSW index serves the
--vector half; this serves the lexical half, which matters more for short strings:
--cosine cannot reliably separate "S. Chen" from "S. Chan", and an exact match
--should win outright rather than depend on the embedding.
CREATE INDEX ON public.surfaces
    USING gin (surface extensions.gin_trgm_ops);
