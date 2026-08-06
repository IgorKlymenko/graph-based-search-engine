CREATE TABLE public.slack_users (
    --'slack' is fixed rather than free: the FK below has to name a source, and a
    --column that can only hold one value is the honest way to write a composite
    --key into a table that is keyed by (source, external_id).
    source          TEXT NOT NULL DEFAULT 'slack' CHECK (source = 'slack'),
    slack_team_id   TEXT NOT NULL,
    slack_user_id   TEXT NOT NULL,
    user_id         UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (source, slack_team_id)
        REFERENCES public.connector_installations (source, external_id)
        ON DELETE CASCADE,
    PRIMARY KEY (slack_team_id, slack_user_id) --keys are unique only within a workspace
);

CREATE INDEX ON public.slack_users (user_id);

ALTER TABLE public.slack_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.slack_users FORCE ROW LEVEL SECURITY;
