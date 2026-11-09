"""The query path: a question in, a grounded answer (or a better question) out.

    retrieve.py   seeds from surfaces, walks PPR over PMI co-occurrence, collects
                  passages; runs inside one tenant-walled transaction
    answer.py     decides a stance -- answer, reframe, clarify -- then answers;
                  renders for a terminal or for Slack

Entry points: ``retrieve.answer_question`` (the Slack worker), and
``python -m src.query.retrieve`` for the CLI (``embed`` and ``ask``).
Supersedes ``old_server/entity_linking/{ask,answer}.py``.
"""
