"""Knowledge-graph extraction pipeline.

Stages: candidate extraction -> candidate embedding -> HDBSCAN clustering + LLM
resolution -> entity query vectors -> relation extraction -> graph upload.
Run with ``uv run python -m pipeline.run --notes-dir <dir>``.
"""
