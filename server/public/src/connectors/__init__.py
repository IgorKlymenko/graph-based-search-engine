"""Connectors: source systems rendered into documents the ingest can read.

    src/connectors/base.py      Document, Mention, the Connector protocol, sync()
    src/connectors/code/        code -> symbols -> text with pre-resolved mentions
    src/connectors/github/      the GitHub client and connector (repos -> documents)

Every connector ends in ``base.sync``: one door into the store, so file
identity, render versions, episode diffing and seeded mentions are paid for
once. The Slack connector predates this package and still lives in
``old_server/entity_linking/slack.py``; it fits the protocol and moves when it
is packaged.
"""
