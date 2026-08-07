"""Tenant isolation, exercised as the server actually connects.

Every test runs the statements a worker actually issues: assume a role, declare
a context, query. Assertions are made from *inside* the wall; seeding is done as
the owner, outside it.

Two roles, split by how the tenant is established rather than by connector:

    app_connector   derives it from a verified installation (app.connector +
                    app.workspace_id). Cannot claim a tenant at all.
    app_ingest      claims it outright (app.tenant_id), for uploads and
                    backfills that have no workspace to derive from.

Skipped unless a Postgres is reachable. Point ``TEST_DATABASE_URL`` at one, or
run ``supabase start`` and the local default is used.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
)
ACME = "aaaaaaaa-0000-4000-8000-00000000000a"
GLOBEX = "bbbbbbbb-0000-4000-8000-00000000000b"
LIVE_TEAM = "T_TEST_LIVE"      # installed, linked to ACME
GONE_TEAM = "T_TEST_GONE"      # linked to GLOBEX but uninstalled
NOTION_WS = "W_TEST_NOTION"    # a second source, same trust model, no new role

WALLED = ["files", "episodes", "surfaces", "entities", "chains", "chain_surfaces",
          "tenants", "memberships", "subscriptions"]
UNWALLED = ["users", "slack_users", "connector_installations"]


@pytest.fixture(scope="module")
def connection():
    try:
        conn = psycopg.connect(DB_URL, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"no database at {DB_URL}: {exc}")
    with conn.cursor() as cur:
        cur.execute("SELECT to_regrole('app_user') IS NOT NULL")
        if not cur.fetchone()[0]:
            pytest.skip("app_user role not present; migrations not applied")
    yield conn
    conn.close()


@pytest.fixture(scope="module", autouse=True)
def seed(connection):
    """Two tenants with a file each, written as the owner (outside the wall)."""
    with connection.cursor() as cur:
        for tenant, name in ((ACME, "Acme"), (GLOBEX, "Globex")):
            cur.execute(
                "INSERT INTO tenants (id, name) VALUES (%s, %s)"
                " ON CONFLICT (id) DO NOTHING", (tenant, name))
            cur.execute(
                "INSERT INTO files (id, tenant_id, source, external_id,"
                " full_text_hash, render_version) VALUES (%s,%s,'upload',%s,%s,1)"
                " ON CONFLICT (id) DO NOTHING",
                (str(uuid.uuid5(uuid.NAMESPACE_DNS, tenant)), tenant,
                 f"{name}.md", "hash"))
        cur.execute(
            "INSERT INTO connector_installations"
            " (source, external_id, tenant_id, token_ref)"
            " VALUES ('slack',%s,%s,'slack-token-ref')"
            " ON CONFLICT DO NOTHING", (LIVE_TEAM, ACME))
        cur.execute(
            "INSERT INTO connector_installations"
            " (source, external_id, tenant_id, token_ref, uninstalled_at)"
            " VALUES ('slack',%s,%s,'gone-ref', now())"
            " ON CONFLICT DO NOTHING", (GONE_TEAM, GLOBEX))
        cur.execute(
            "INSERT INTO connector_installations"
            " (source, external_id, tenant_id, token_ref)"
            " VALUES ('notion',%s,%s,'notion-token-ref')"
            " ON CONFLICT DO NOTHING", (NOTION_WS, GLOBEX))
    yield
    with connection.cursor() as cur:
        cur.execute("DELETE FROM connector_installations"
                    " WHERE external_id = ANY(%s)",
                    ([LIVE_TEAM, GONE_TEAM, NOTION_WS],))
        cur.execute("DELETE FROM files WHERE tenant_id = ANY(%s)", ([ACME, GLOBEX],))
        cur.execute("DELETE FROM tenants WHERE id = ANY(%s)", ([ACME, GLOBEX],))


def run(connection, sql, params=None, *, role="app_ingest", tenant=None,
        connector=None, workspace=None):
    """One transaction shaped like a request: assume the role, declare, query.

    SET LOCAL throughout. On a transaction-pooled connection a session-level SET
    outlives the request and the next caller inherits the previous tenant, which
    is the failure this whole design exists to prevent.
    """
    with connection.transaction():
        cur = connection.cursor()
        cur.execute(f"SET LOCAL ROLE {role}")
        for key, value in (("app.tenant_id", tenant),
                           ("app.connector", connector),
                           ("app.workspace_id", workspace)):
            if value is not None:
                cur.execute("SELECT set_config(%s, %s, true)", (key, value))
        cur.execute(sql, params)
        return cur.fetchall() if cur.description else None


# ── the wall, exercised through the ingest role (a bare tenant claim) ────────

def test_sees_only_the_declared_tenant(connection):
    rows = run(connection, "SELECT tenant_id FROM files", tenant=ACME)
    assert rows, "the declared tenant's own rows must be visible"
    assert {r[0] for r in rows} == {uuid.UUID(ACME)}


def test_the_other_tenant_is_invisible_not_merely_filtered(connection):
    """Naming another tenant explicitly must still return nothing."""
    rows = run(connection, "SELECT count(*) FROM files WHERE tenant_id = %s",
               (GLOBEX,), tenant=ACME)
    assert rows[0][0] == 0


def test_no_context_means_no_rows(connection):
    """Fail-closed: an unset context denies rather than exposing."""
    for table in WALLED:
        rows = run(connection, f"SELECT count(*) FROM {table}")
        assert rows[0][0] == 0, f"{table} leaked rows with no tenant context"


def test_cannot_write_into_another_tenant(connection):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection,
            "INSERT INTO files (id, tenant_id, source, external_id,"
            " full_text_hash, render_version)"
            " VALUES (gen_random_uuid(), %s, 'upload', 'x.md', 'h', 1)",
            (GLOBEX,), tenant=ACME)


def test_cannot_move_a_row_to_another_tenant(connection):
    """WITH CHECK must block the update even though USING admits the row."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection, "UPDATE files SET tenant_id = %s WHERE tenant_id = %s",
            (GLOBEX, ACME), tenant=ACME)


def test_cannot_delete_another_tenants_rows(connection):
    run(connection, "DELETE FROM files WHERE tenant_id = %s", (GLOBEX,),
        tenant=ACME)
    with connection.cursor() as cur:  # as owner: the row survived
        cur.execute("SELECT count(*) FROM files WHERE tenant_id = %s", (GLOBEX,))
        assert cur.fetchone()[0] == 1


def test_writes_into_the_declared_tenant_are_allowed(connection):
    run(connection,
        "INSERT INTO files (id, tenant_id, source, external_id,"
        " full_text_hash, render_version)"
        " VALUES (gen_random_uuid(), %s, 'upload', 'allowed.md', 'h', 1)",
        (ACME,), tenant=ACME)
    rows = run(connection,
               "SELECT count(*) FROM files WHERE external_id = 'allowed.md'",
               tenant=ACME)
    assert rows[0][0] == 1


# ── the connector role: derives, never claims ────────────────────────────────

def test_the_workspace_resolves_its_own_tenant(connection):
    """The worker names a workspace; the database decides which tenant that is."""
    rows = run(connection, "SELECT tenant_id FROM files",
               role="app_connector", connector="slack", workspace=LIVE_TEAM)
    assert {r[0] for r in rows} == {uuid.UUID(ACME)}


def test_a_connector_cannot_claim_a_tenant(connection):
    """The reason the roles are split.

    With one role holding both paths, a worker that set app.tenant_id instead of
    app.workspace_id would silently fall back to a bare claim. Here the resolver
    never reads app.tenant_id for this role, so the mistake denies instead.
    """
    rows = run(connection, "SELECT count(*) FROM files",
               role="app_connector", tenant=ACME)
    assert rows[0][0] == 0


def test_an_uninstalled_workspace_grants_nothing(connection):
    """Removing the app has to stop access, not merely stop new events."""
    rows = run(connection, "SELECT count(*) FROM files",
               role="app_connector", connector="slack", workspace=GONE_TEAM)
    assert rows[0][0] == 0


def test_an_unknown_workspace_denies_even_with_a_tenant_also_declared(connection):
    """A stale workspace id must not fall through to whatever else was set."""
    rows = run(connection, "SELECT count(*) FROM files",
               role="app_connector", connector="slack",
               workspace="W_DOES_NOT_EXIST", tenant=ACME)
    assert rows[0][0] == 0


def test_the_source_is_part_of_the_key(connection):
    """A workspace id from one source must not resolve under another."""
    rows = run(connection, "SELECT count(*) FROM files",
               role="app_connector", connector="notion", workspace=LIVE_TEAM)
    assert rows[0][0] == 0


def test_a_second_source_needs_no_new_role_or_policy(connection):
    """Adding Notion is a row in connector_installations, nothing more."""
    rows = run(connection, "SELECT tenant_id FROM files",
               role="app_connector", connector="notion", workspace=NOTION_WS)
    assert {r[0] for r in rows} == {uuid.UUID(GLOBEX)}


def test_connector_writes_are_walled_to_its_workspace(connection):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection,
            "INSERT INTO files (id, tenant_id, source, external_id,"
            " full_text_hash, render_version)"
            " VALUES (gen_random_uuid(), %s, 'slack', 'sneak.md', 'h', 1)",
            (GLOBEX,), role="app_connector", connector="slack",
            workspace=LIVE_TEAM)


# ── the token reference ──────────────────────────────────────────────────────

def test_the_worker_gets_only_its_own_token_ref(connection):
    rows = run(connection, "SELECT public.current_token_ref()",
               role="app_connector", connector="slack", workspace=LIVE_TEAM)
    assert rows[0][0] == "slack-token-ref"


def test_no_workspace_means_no_token_ref(connection):
    rows = run(connection, "SELECT public.current_token_ref()",
               role="app_connector")
    assert rows[0][0] is None


def test_an_uninstalled_workspace_yields_no_token_ref(connection):
    rows = run(connection, "SELECT public.current_token_ref()",
               role="app_connector", connector="slack", workspace=GONE_TEAM)
    assert rows[0][0] is None


def test_installations_cannot_be_read_directly(connection):
    """A SELECT grant here would expose every workspace and its tenant."""
    for role in ("app_connector", "app_ingest"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            run(connection, "SELECT count(*) FROM connector_installations",
                role=role, connector="slack", workspace=LIVE_TEAM)


def test_ingest_cannot_reach_the_token_ref(connection):
    """Only the role that proves a workspace may fetch its credential."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection, "SELECT public.current_token_ref()",
            role="app_ingest", tenant=ACME)


# ── the surfaces of the roles themselves ─────────────────────────────────────

def test_identity_tables_are_out_of_reach(connection):
    """No tenant column means no wall, so the grant is revoked instead."""
    for table in UNWALLED:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            run(connection, f"SELECT count(*) FROM {table}", tenant=ACME)


def test_tenancy_and_billing_are_read_only(connection):
    for table in ("tenants", "memberships", "subscriptions"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            run(connection, f"DELETE FROM {table}", tenant=ACME)


def test_the_role_cannot_bypass_rls(connection):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection, "ALTER TABLE files DISABLE ROW LEVEL SECURITY",
            tenant=ACME)


def test_the_role_cannot_create_objects(connection):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection, "CREATE TABLE public.evil (x int)", tenant=ACME)


def test_context_does_not_survive_the_transaction(connection):
    """SET LOCAL, not SET: a pooled connection must not inherit a tenant."""
    run(connection, "SELECT 1", tenant=ACME)
    rows = run(connection,
               "SELECT coalesce(current_setting('app.tenant_id', true), '')")
    assert rows[0][0] == "", "tenant context leaked past the transaction"


def test_the_roles_carry_a_statement_timeout_for_when_they_can_log_in(connection):
    """Asserts the setting is configured, not that it is in force.

    ALTER ROLE ... SET applies at *login*, against the role that authenticated.
    These roles are NOLOGIN and reached by SET ROLE, so the ceiling is inert
    until the server connects as a dedicated LOGIN role granted one of them --
    checking `SHOW statement_timeout` here would assert something that cannot be
    true by design. Until then the limit has to come from the connection string
    or a SET LOCAL beside the SET LOCAL ROLE.
    """
    with connection.cursor() as cur:
        cur.execute(
            "SELECT rolname, coalesce(array_to_string(rolconfig, ' '), '')"
            " FROM pg_roles WHERE rolname = ANY(%s)",
            (["app_connector", "app_ingest", "app_user"],))
        for role, config in cur.fetchall():
            assert "statement_timeout" in config, f"{role} has no ceiling set"
            assert "idle_in_transaction_session_timeout" in config, role


# ── the login roles: where the ceilings and the RLS floor actually land ───────

LOGIN_PASSWORD = "localdev-only"


@pytest.fixture(scope="module")
def login_url(connection):
    """A connection string for app_connector_login.

    The migration creates it without a password on purpose -- a password in a
    migration lands in git and in the statement log. Setting one here is a local
    test convenience; production sets it from the secret store.
    """
    with connection.cursor() as cur:
        cur.execute("SELECT to_regrole('app_connector_login') IS NOT NULL")
        if not cur.fetchone()[0]:
            pytest.skip("login roles not present")
        cur.execute(f"ALTER ROLE app_connector_login PASSWORD '{LOGIN_PASSWORD}'")
    parsed = DB_URL.split("@", 1)[1]
    return f"postgresql://app_connector_login:{LOGIN_PASSWORD}@{parsed}"


def test_the_ceiling_applies_at_login_where_set_role_could_not(login_url):
    """The whole reason these roles exist.

    ALTER ROLE ... SET is applied at login. Set on a NOLOGIN role reached by
    SET ROLE it reads back as 0 -- the limit silently does not exist. Set on the
    role that authenticates, it holds, and SET ROLE afterwards does not reset it.
    """
    with psycopg.connect(login_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SHOW statement_timeout")
        assert cur.fetchone()[0] == "30s"
        cur.execute("SHOW idle_in_transaction_session_timeout")
        assert cur.fetchone()[0] == "10s"


def test_the_login_role_does_not_bypass_rls(login_url):
    """postgres carries BYPASSRLS, so a forgotten SET ROLE there sees everything.

    The floor here is a role that cannot bypass anything.
    """
    with psycopg.connect(login_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user")
        assert cur.fetchone()[0] is False


def test_forgetting_set_role_fails_loudly(login_url):
    """NOINHERIT is what makes the mistake safe.

    With INHERIT the login role would hold app_connector's privileges
    automatically: SET ROLE becomes optional, current_setting('role') stays
    'none', and the resolver -- which reads it to tell ingest from connector --
    would misresolve instead of denying. Here the role can do nothing but
    connect until it names one.
    """
    with psycopg.connect(login_url, autocommit=True) as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("SELECT count(*) FROM files")


def test_after_set_role_it_works_and_keeps_the_ceiling(login_url, connection):
    with connection.cursor() as cur:  # seed as owner
        cur.execute("INSERT INTO connector_installations"
                    " (source, external_id, tenant_id, token_ref)"
                    " VALUES ('slack', %s, %s, 'ref') ON CONFLICT DO NOTHING",
                    (LIVE_TEAM, ACME))
    with psycopg.connect(login_url) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL ROLE app_connector")
        cur.execute("SELECT set_config('app.connector', 'slack', true)")
        cur.execute("SELECT set_config('app.workspace_id', %s, true)", (LIVE_TEAM,))
        cur.execute("SELECT tenant_id FROM files")
        assert {r[0] for r in cur.fetchall()} == {uuid.UUID(ACME)}
        cur.execute("SHOW statement_timeout")
        assert cur.fetchone()[0] == "30s", "SET ROLE must not reset the ceiling"


# ── internal.events: dedupe at ACK, idempotent claim at /work ────────────────

EVENT = "Ev_TEST_1"


@pytest.fixture(scope="module", autouse=True)
def clean_events(connection):
    yield
    with connection.cursor() as cur:
        cur.execute("DELETE FROM internal.events WHERE external_id = ANY(%s)",
                    ([LIVE_TEAM, GONE_TEAM],))


def ack(connection, event_id, workspace=LIVE_TEAM, retry=0):
    """The ACK path: one insert, and a rowcount of zero means 'already seen'."""
    with connection.transaction():
        cur = connection.cursor()
        cur.execute("SET LOCAL ROLE app_ack")
        cur.execute("SELECT set_config('app.connector', 'slack', true)")
        cur.execute("SELECT set_config('app.workspace_id', %s, true)", (workspace,))
        cur.execute(
            "INSERT INTO internal.events (source, external_id, event_id,"
            " payload, retry_num) VALUES ('slack', %s, %s, %s, %s)"
            " ON CONFLICT DO NOTHING",
            (workspace, event_id, '{"t":"x"}', retry))
        return cur.rowcount


def claim(connection, event_id, workspace=LIVE_TEAM):
    """The worker path: one conditional update, atomic, no lock."""
    with connection.transaction():
        cur = connection.cursor()
        cur.execute("SET LOCAL ROLE app_connector")
        cur.execute("SELECT set_config('app.connector', 'slack', true)")
        cur.execute("SELECT set_config('app.workspace_id', %s, true)", (workspace,))
        cur.execute(
            "UPDATE internal.events SET status = 'processing', started_at = now()"
            " WHERE source = 'slack' AND external_id = %s AND event_id = %s"
            "   AND status IN ('queued','failed') RETURNING payload",
            (workspace, event_id))
        return cur.fetchall()


def test_a_slack_retry_is_deduped_by_the_primary_key(connection):
    """Idempotency as a constraint rather than application discipline."""
    assert ack(connection, EVENT) == 1
    assert ack(connection, EVENT, retry=1) == 0, "the retry created a second row"


def test_the_claim_is_idempotent_under_duplicate_delivery(connection):
    """Cloud Tasks will deliver twice; only one worker may take the work."""
    ack(connection, "Ev_claim")
    assert len(claim(connection, "Ev_claim")) == 1
    assert claim(connection, "Ev_claim") == [], "a second worker also claimed it"


def test_the_ack_role_cannot_reach_the_corpus(connection):
    """The whole reason it is not a member of app_user.

    The ACK endpoint is publicly reachable because Slack posts to it. A bug
    there running as app_connector could read the workspace's entire corpus.
    """
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection, "SELECT count(*) FROM files",
            role="app_ack", connector="slack", workspace=LIVE_TEAM)


def test_the_ack_role_cannot_even_read_events_back(connection):
    """Insert-only: the dedupe needs a rowcount, never a row."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection, "SELECT count(*) FROM internal.events",
            role="app_ack", connector="slack", workspace=LIVE_TEAM)


def test_the_worker_cannot_invent_an_event(connection):
    """An event that was never ACKed must not appear from the work side."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run(connection,
            "INSERT INTO internal.events (source, external_id, event_id)"
            " VALUES ('slack', %s, 'Ev_forged')", (LIVE_TEAM,),
            role="app_connector", connector="slack", workspace=LIVE_TEAM)


def test_events_are_walled_by_workspace_not_tenant(connection):
    """The tenant is unknown at ACK time, so the workspace is the isolation key."""
    ack(connection, "Ev_walled")
    rows = run(connection, "SELECT count(*) FROM internal.events",
               role="app_connector", connector="slack", workspace=GONE_TEAM)
    assert rows[0][0] == 0


def test_events_deny_with_no_workspace_declared(connection):
    rows = run(connection, "SELECT count(*) FROM internal.events",
               role="app_connector")
    assert rows[0][0] == 0
