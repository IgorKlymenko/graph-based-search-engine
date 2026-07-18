"""Knowledge-graph extraction pipeline.

Stages: mention extraction -> mention embedding -> HDBSCAN clustering + LLM
resolution -> entity query vectors -> relation extraction -> graph upload.
Run with ``uv run python -m pipeline.run --notes-dir <dir>``.
"""
