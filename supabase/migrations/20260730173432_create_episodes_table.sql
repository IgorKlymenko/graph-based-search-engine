CREATE TABLE public.episodes (
    id              UUID PRIMARY KEY,
    file_id         UUID NOT NULL,
    tenant_id       UUID NOT NULL,
    text            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expired_at      TIMESTAMPTZ, --removing file invalidates all the episodes on canscade; but the episode can be still invalidated searately if file changes from the code
    FOREIGN KEY (file_id, tenant_id) 
        REFERENCES public.files (id, tenant_id) ON DELETE CASCADE,
    UNIQUE (tenant_id, id), --for the later reference in the nodes using the composite FK
    CHECK (expired_at IS NULL OR expired_at >= created_at)
);


--serves the cascade trigger and "all episodes of this file" during sync
CREATE INDEX ON public.episodes (file_id);

--live-episode lookups per tenant
CREATE INDEX ON public.episodes (tenant_id) WHERE expired_at IS NULL; -- some queries will land via file_id or id, but it worth just checking in on this later when there is logic for that. This is for RLS I guess>?

ALTER TABLE public.episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.episodes FORCE ROW LEVEL SECURITY;

CREATE FUNCTION cascade_file_removal() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE public.episodes
       SET expired_at = NEW.removed_at
     WHERE file_id = NEW.id
       AND expired_at IS NULL;
    RETURN NEW;
END $$;

CREATE TRIGGER cascade_removal
    AFTER UPDATE ON public.files
    FOR EACH ROW
    WHEN (OLD.removed_at IS NULL AND NEW.removed_at IS NOT NULL)
    EXECUTE FUNCTION cascade_file_removal();