"""Entity linking, packaged as one call: documents in, graph rows out.

    from src.inject.main import ingest
    ingest(notes_dir=Path("/tmp/corpus"), tenant_id=..., dsn=...)

Stages run in filename order under ``src/``:

    1_episodes.py   split documents into episodes; detect what changed
    2_chains.py     span inventory + one LLM call per episode -> coref chains
    3_encode.py     one forward pass per episode -> a vector per chain
    4_link.py       score chain pairs, review the uncertain ones, cluster
    5_store.py      write the result to Postgres, in one transaction

Python cannot import a module whose name starts with a digit, so ``_stages``
loads them by path and registers them here under ordinary names. Nothing outside
that file deals with the numbers.
"""

from ._stages import load

episodes = load("1_episodes", "src.inject.episodes")
chains = load("2_chains", "src.inject.chains")
encoding = load("3_encode", "src.inject.encoding")
pairing = load("4_link", "src.inject.pairing")
store = load("5_store", "src.inject.store")

__all__ = ["chains", "encoding", "episodes", "pairing", "store"]
