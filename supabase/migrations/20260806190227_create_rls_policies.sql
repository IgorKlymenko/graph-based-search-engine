-- Every policy, in one place. Two layers, and the distinction is the design:
--
--   RESTRICTIVE walls  say *whose* rows. They AND with every permissive policy,
--                      so no door can admit a row from another tenant.
--   PERMISSIVE doors   say *what* the role may do. They carry no tenant
--                      predicate at all -- the walls own that.
--
-- Both scoped TO app_user, which app_connector and app_ingest are members of, so
-- policies naming app_user bind all three. When `authenticated` gains direct
-- access it needs its own walls over current_tenant_ids(): RESTRICTIVE policies
-- bind only the roles they name, and a role with no wall has no isolation.
--
-- The wall compares against a single tenant, not a set. An unset context makes
-- current_tenant_id() NULL, `tenant_id = NULL` is NULL rather than true, and the
-- wall denies. Fail-closed is the point -- forgetting to declare a tenant locks
-- the server out rather than opening the corpus up.

-- ── restrictive walls ────────────────────────────────────────────────────────

CREATE POLICY tenants_tenant_wall ON public.tenants  --its own id IS the tenant column
    AS RESTRICTIVE FOR ALL TO app_user
    USING (id = public.current_tenant_id())
    WITH CHECK (id = public.current_tenant_id());

CREATE POLICY memberships_tenant_wall ON public.memberships
    AS RESTRICTIVE FOR ALL TO app_user
    USING (tenant_id = public.current_tenant_id())
    WITH CHECK (tenant_id = public.current_tenant_id());

CREATE POLICY subscriptions_tenant_wall ON public.subscriptions
    AS RESTRICTIVE FOR ALL TO app_user
    USING (tenant_id = public.current_tenant_id())
    WITH CHECK (tenant_id = public.current_tenant_id());

CREATE POLICY files_tenant_wall ON public.files
    AS RESTRICTIVE FOR ALL TO app_user
    USING (tenant_id = public.current_tenant_id())
    WITH CHECK (tenant_id = public.current_tenant_id());

CREATE POLICY episodes_tenant_wall ON public.episodes
    AS RESTRICTIVE FOR ALL TO app_user
    USING (tenant_id = public.current_tenant_id())
    WITH CHECK (tenant_id = public.current_tenant_id());

CREATE POLICY surfaces_tenant_wall ON public.surfaces
    AS RESTRICTIVE FOR ALL TO app_user
    USING (tenant_id = public.current_tenant_id())
    WITH CHECK (tenant_id = public.current_tenant_id());

CREATE POLICY entities_tenant_wall ON public.entities
    AS RESTRICTIVE FOR ALL TO app_user
    USING (tenant_id = public.current_tenant_id())
    WITH CHECK (tenant_id = public.current_tenant_id());

CREATE POLICY chains_tenant_wall ON public.chains
    AS RESTRICTIVE FOR ALL TO app_user
    USING (tenant_id = public.current_tenant_id())
    WITH CHECK (tenant_id = public.current_tenant_id());

CREATE POLICY chain_surfaces_tenant_wall ON public.chain_surfaces
    AS RESTRICTIVE FOR ALL TO app_user
    USING (tenant_id = public.current_tenant_id())
    WITH CHECK (tenant_id = public.current_tenant_id());


-- No walls on users / slack_users / connector_installations: identity
-- resolution runs before a tenant is known, so a tenant predicate could never be
-- satisfied there. Their grants are revoked in the roles migration instead --
-- privilege denial raises, whereas a missing policy silently returns nothing.

-- ── permissive doors ─────────────────────────────────────────────────────────

-- Content: full access, walled to the resolved tenant.
CREATE POLICY files_app_rw           ON public.files          FOR ALL TO app_user USING (true) WITH CHECK (true);
CREATE POLICY episodes_app_rw        ON public.episodes       FOR ALL TO app_user USING (true) WITH CHECK (true);
CREATE POLICY surfaces_app_rw        ON public.surfaces       FOR ALL TO app_user USING (true) WITH CHECK (true);
CREATE POLICY entities_app_rw        ON public.entities       FOR ALL TO app_user USING (true) WITH CHECK (true);
CREATE POLICY chains_app_rw          ON public.chains         FOR ALL TO app_user USING (true) WITH CHECK (true);
CREATE POLICY chain_surfaces_app_rw  ON public.chain_surfaces FOR ALL TO app_user USING (true) WITH CHECK (true);

-- Read-only: writes to tenancy and billing go through the privileged path. The
-- revoked grants are what enforce that; these only open the read.
CREATE POLICY tenants_app_read       ON public.tenants        FOR SELECT TO app_user USING (true);
CREATE POLICY memberships_app_read   ON public.memberships    FOR SELECT TO app_user USING (true);
CREATE POLICY subscriptions_app_read ON public.subscriptions  FOR SELECT TO app_user USING (true);
