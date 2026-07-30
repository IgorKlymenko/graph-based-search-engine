CREATE TABLE public.files (
    id              UUID PRIMARY KEY, --UUIDv5 injested using (tenant_id, source, external_id) in code
    tenant_id       UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    source          TEXT NOT NULL CHECK (source IN ('slack','gdrive','notion','upload')), --original provider
    external_id     TEXT NOT NULL, -- id within the provider
    title           TEXT, --later on when citing in the prompt
    url             TEXT, --later on when directing ppl in the prompt
    full_text_hash  TEXT NOT NULL, --hash of the whole file to check during the sync
    render_version  INT NOT NULL,  --my renderer from slack/notion to text API version
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now(), --write it on every check from the app, including the unchanged path
    changed_at      TIMESTAMPTZ NOT NULL DEFAULT now(), --the time file was changed
    removed_at      TIMESTAMPTZ, --if during sync there is no file it is removed and it's episodes/nodes are also removed (have to happen in the SAME TRANSACTION)
    UNIQUE (tenant_id, source, external_id), --its id is hashed on these three; this is backup if code changes the hashing somehow
    UNIQUE (id, tenant_id) --to be used as the composite Foreign Key in nodes/episodes referencing
);

CREATE INDEX ON public.files (tenant_id) WHERE removed_at IS NULL; --to search over non-removed files

ALTER TABLE public.files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.files FORCE ROW LEVEL SECURITY;

CREATE TRIGGER set_changed_at
    BEFORE UPDATE ON public.files
    FOR EACH ROW
    WHEN (OLD.full_text_hash IS DISTINCT FROM NEW.full_text_hash --to update ONLY when the context of the file has changed (not during the sync changes)
       OR OLD.render_version IS DISTINCT FROM NEW.render_version)
    EXECUTE FUNCTION extensions.moddatetime(changed_at);