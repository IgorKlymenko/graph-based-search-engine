-- Every function the policies call, in one place.
--
-- Two access models live here and must not be confused:
--
--   current_tenant_id()   the SERVER's tenant. Derived from a verified
--                         installation, or claimed outright by the ingest role.
--                         This is what gates the app_* roles today.
--   current_tenant_ids()  an END USER's tenants, from their JWT via auth.uid().
--                         Nothing uses it yet -- it is here for when
--                         `authenticated` gets direct access.
--
-- A server connecting on its own Postgres role has no JWT, so auth.uid() is
-- NULL there and the membership helpers return nothing. Gating the server on
-- them denies every row and every write -- which is exactly what happened
-- before this split existed.

-- Which branch runs is decided by *who is asking*, not by which settings happen
-- to be present. A worker cannot reach the weaker path by setting a different
-- variable, only by being a different role -- a deployment decision rather than
-- a line of code.
--
--   app_ingest      claims a tenant outright. Uploads and backfills, which have
--                   no workspace to derive from.
--   anything else   derives it from a verified installation, or gets NULL.
--
-- Deriving is the default deliberately: a role added later, or app_user assumed
-- directly, lands on the strict branch. Reaching the claim requires naming
-- app_ingest.
--
-- current_setting('role'), NOT current_user: inside a SECURITY DEFINER function
-- current_user is the *owner*, so the branch would never fire and every ingest
-- request would silently resolve to NULL. session_user is no better -- it is the
-- login role, not the one SET ROLE selected.
--
-- SECURITY DEFINER because the callers have no grant on connector_installations,
-- deliberately. That read works only because the owner (postgres) has BYPASSRLS:
-- the table is FORCE ROW LEVEL SECURITY, which strips the owner's usual
-- exemption, so a definer function owned by a role WITHOUT BYPASSRLS reads zero
-- rows, returns NULL, and silently denies everything. If this is ever
-- reassigned, check that first -- the failure is total and reads as a
-- configuration problem rather than an ownership one.
CREATE FUNCTION public.current_tenant_id() RETURNS uuid
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = '' AS $$
    SELECT CASE
        WHEN current_setting('role', true) = 'app_ingest'
        THEN nullif(current_setting('app.tenant_id', true), '')::uuid
        ELSE (
            SELECT i.tenant_id FROM public.connector_installations i
             WHERE i.source = nullif(current_setting('app.connector', true), '')
               AND i.external_id =
                   nullif(current_setting('app.workspace_id', true), '')
               AND i.uninstalled_at IS NULL
        )
    END;
$$;

-- The worker needs its bot token to call the source back. This returns the one
-- row it has already proven it is acting for, and nothing else -- a plain SELECT
-- grant on the table would let any worker enumerate every workspace and tenant.
--
-- Returns the *reference*, never a secret: the token lives in the secret store.
-- A database that never holds the token cannot leak it.
CREATE FUNCTION public.current_token_ref() RETURNS text
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = '' AS $$
    SELECT i.token_ref FROM public.connector_installations i
     WHERE i.source = nullif(current_setting('app.connector', true), '')
       AND i.external_id = nullif(current_setting('app.workspace_id', true), '')
       AND i.uninstalled_at IS NULL;
$$;

-- End-user helpers. Unused until `authenticated` gets direct access; correct for
-- that model, wrong for the server, which is why they gate nothing today.
--
-- STABLE so the planner evaluates once per query rather than per row.
-- SECURITY DEFINER because the caller cannot read memberships directly -- the
-- wall on that table would recurse into this very function.
-- search_path is emptied and every name qualified: a SECURITY DEFINER function
-- resolving names through a caller-controlled path is a privilege hole.
CREATE FUNCTION public.current_tenant_ids() RETURNS SETOF uuid
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = '' AS $$
    SELECT tenant_id FROM public.memberships
    WHERE user_id = auth.uid();
$$;

CREATE FUNCTION public.has_role_in(t uuid, required text[]) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = '' AS $$
    SELECT EXISTS (SELECT 1 FROM public.memberships
                   WHERE user_id = auth.uid()
                     AND tenant_id = t
                     AND role = ANY(required));
$$;

-- SECURITY DEFINER functions are executable by PUBLIC unless revoked. Every one
-- of these reads a table its callers cannot.
REVOKE EXECUTE ON FUNCTION public.current_tenant_id() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.current_token_ref() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.current_tenant_ids() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.has_role_in(uuid, text[]) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.current_tenant_ids() TO authenticated;
GRANT EXECUTE ON FUNCTION public.has_role_in(uuid, text[]) TO authenticated;
-- app_* grants live with the roles, in the next migration.
