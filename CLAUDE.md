# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Core Development Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum

3. Testing Requirements
   - Framework: `uv run pytest`
   - Async testing: use anyio, not asyncio
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

- For commits fixing bugs or adding features based on user reports add:
  ```bash
  git commit --trailer "Reported-by:<name>"
  ```
  Where `<name>` is the name of the user.

- For commits related to a Github issue, add
  ```bash
  git commit --trailer "Github-Issue:#<number>"
  ```
- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## System Architecture

[fill in here]

## Core Components

- `config.py`: Configuration management
- `daemon.py`: Main daemon
[etc... fill in here]

## Database access — non-negotiable rules for any client

The database enforces tenant isolation, but only if the client sets up each
request correctly. These are the parts the schema cannot enforce for you.
Isolation is verified by `old_server/tests/test_db_tenant_isolation.py` (36
tests); reasoning is in `ADRs/supabase/security.md`.

### Every request is one transaction, in this shape

```python
async with pool.connection() as conn:          # logged in as app_*_login
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE app_connector")
        await conn.execute(
            "SELECT set_config('app.connector', %s, true)", (source,))
        await conn.execute(
            "SELECT set_config('app.workspace_id', %s, true)", (team_id,))
        ...                                     # real work here
```

**`SET LOCAL`, never `SET`.** Both the role and the context. A session-level
setting outlives the transaction, and on a pooled connection the next request
inherits the previous caller's tenant. This is the single worst bug available
here: it works in dev (session pooling) and leaks in production (transaction
pooling).

**One transaction.** The role, the context and the queries. Context dies at
COMMIT by design — reusing a connection without re-declaring is a denied query,
not a stale one.

**Parameterise `set_config`.** The workspace id arrives from an external system.
Never interpolate it into SQL.

### Pick the right role

| Role | Use | Declares |
|---|---|---|
| `app_connector` | anything with a workspace (Slack, later Drive/Notion) | `app.connector`, `app.workspace_id` |
| `app_ingest` | uploads, backfills — no workspace to derive from | `app.tenant_id` |
| `app_ack` | the public webhook endpoint, nothing else | `app.connector`, `app.workspace_id` |

`app_connector` **cannot** claim a tenant — setting `app.tenant_id` on it
resolves to NULL and every query returns nothing. That is deliberate: the
database resolves the tenant from `connector_installations`, so a wrong tenant
computed in application code is caught rather than honoured.

`app_ack` cannot read the corpus, or even read events back. Its whole job is one
insert.

**Never connect as `postgres`.** It carries `BYPASSRLS`, so nothing is walled.
The login roles do not, which is the point of them.

**Login roles are created without a password** — set it from the secret store,
never in a migration. Forgetting `SET ROLE` fails loudly (the login roles are
`NOINHERIT`); do not "fix" that by granting INHERIT.

### Write ordering

Extraction output must land in **one transaction**, in FK order:

```
surfaces  →  chains  →  chain_surfaces
```

A partial write leaves chains with no surfaces, and the evidence queries then
return *silently wrong* answers — `alias_hit` simply says no. Failing loudly is
not the default here; correctness depends on atomicity.

### Webhook handling

ACK path — dedupe is the primary key, not a lookup:

```sql
INSERT INTO internal.events (source, external_id, event_id, payload, retry_num)
VALUES (...) ON CONFLICT DO NOTHING;
-- rowcount 0 ⇒ already seen. Return 200, enqueue nothing.
```

Do not `SELECT` first to check — that is a race. Worker path — claim with one
conditional update:

```sql
UPDATE internal.events SET status = 'processing', started_at = now()
 WHERE source = $1 AND external_id = $2 AND event_id = $3
   AND status IN ('queued','failed')
RETURNING payload;
-- rowcount 0 ⇒ another worker has it, or it is done. Stop.
```

Cloud Tasks *will* deliver twice. This is what makes that safe without a lock.

### Reading and writing entities

- Read a name with `public.entity_display_name(tenant, entity)`. Do not read
  `display_name` directly — it is a human override and is usually NULL.
- Write `display_name` **only** when a person corrected it. Re-clustering must
  have nothing to overwrite.
- `chains.entity_id` is nullable. A chain without an entity is normal (not yet
  clustered, or a contested name deliberately withheld), not an error.

### Limits to design around

`statement_timeout` 30s (5s on `app_ack`), `idle_in_transaction_session_timeout`
10s, connection limits 30/10/40. Long operations must be chunked; these are
blast-radius caps, so hitting one should read as a bug rather than as load.

### Still unverified

Behaviour under the **transaction pooler**. `SET LOCAL` is correct for it, but
only direct sessions have been exercised. Verify before the first production
deploy — this is exactly where a tenant leak would appear and not be caught in
dev.

## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

- Always add `ArthurClune` as reviewer.

- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Python Tools

## Code Formatting

1. Ruff
   - Format: `uv run ruff format .`
   - Check: `uv run ruff check .`
   - Fix: `uv run ruff check . --fix`
   - Critical issues:
     - Line length (88 chars)
     - Import sorting (I001)
     - Unused imports
   - Line wrapping:
     - Strings: use parentheses
     - Function calls: multi-line with proper indent
     - Imports: split into multiple lines

2. Type Checking
   - Tool: `uv run pyright`
   - Requirements:
     - Explicit None checks for Optional
     - Type narrowing for strings
     - Version warnings can be ignored if checks pass

3. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Prettier (YAML/JSON), Ruff (Python)
   - Ruff updates:
     - Check PyPI versions
     - Update config rev
     - Commit config first

## Error Resolution

1. CI Failures
   - Fix order:
     1. Formatting
     2. Type errors
     3. Linting
   - Type errors:
     - Get full line context
     - Check Optional types
     - Add type narrowing
     - Verify function signatures

2. Common Issues
   - Line length:
     - Break strings with parentheses
     - Multi-line function calls
     - Split imports
   - Types:
     - Add None checks
     - Narrow string types
     - Match existing patterns

3. Best Practices
   - Check git status before commits
   - Run formatters before type checks
   - Keep changes minimal
   - Follow existing patterns
   - Document public APIs
   - Test thoroughly