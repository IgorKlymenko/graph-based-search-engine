CREATE TABLE public.slack_users (
    slack_team_id    TEXT NOT NULL REFERENCES public.slack_teams(id) ON DELETE CASCADE,
    slack_user_id    TEXT NOT NULL,
    user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (slack_team_id, slack_user_id) --because keys are unique only within a workspace
);

CREATE INDEX ON public.slack_users (user_id);

ALTER TABLE public.slack_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.slack_users FORCE ROW LEVEL SECURITY;
