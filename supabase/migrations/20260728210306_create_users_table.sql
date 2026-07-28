CREATE TABLE public.users (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    auth_user_id    uuid UNIQUE REFERENCES auth.users(id) ON DELETE SET NULL,  -- nullable for slack users
    email           text CHECK (email = lower(email)),   -- nullable
    email_source   text CHECK (email_source IN ('slack', 'oidc')),
    name            TEXT CHECK (length(trim(name)) > 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT email_and_source_together
        CHECK ((email IS NULL) = (email_source IS NULL))
);

CREATE UNIQUE INDEX users_email_key ON public.users (email) WHERE email IS NOT NULL;

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.users FORCE ROW LEVEL SECURITY;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE FUNCTION extensions.moddatetime(updated_at);