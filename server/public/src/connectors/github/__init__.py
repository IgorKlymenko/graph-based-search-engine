"""GitHub: repositories -> one document per source or markdown file.

    export GITHUB_TOKEN=ghp_...
    uv run python -m src.connectors.github --tenant <uuid> --store ./store.json \\
        --repos owner/name [owner/other] [--ref main]
    uv run python -m src.inject.main --store ./store.json --tenant-id <uuid> --dsn ...

``client.py`` is the smallest REST client that does this job; ``connector.py``
turns a repository tree into ``Document``s and remembers, per repo, which blob
it rendered for each path so a re-sync fetches only what changed.
"""
