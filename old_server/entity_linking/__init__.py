"""Entity linking: documents -> episodes -> chains -> vectors -> entities.

Stages run in filename order:

    1_episodes.py   split documents into episodes; detect what changed
    2_chains.py     span inventory + one LLM call per episode -> coref chains
    3_encode.py     one forward pass per episode -> a vector per chain
    4_link.py       score chain pairs, review the uncertain ones, cluster

``pipeline.py`` runs all four. Support modules (``spans``, ``prompts``,
``coref_metrics``) are not stages and keep ordinary names.

See ``RUNNING.md`` for setup, commands and gotchas; ``DATAFLOW.md`` for the
objects that move between stages.
"""

from ._stages import load

episodes = load("1_episodes", "entity_linking.episodes")
chains = load("2_chains", "entity_linking.chains")
encoding = load("3_encode", "entity_linking.encoding")
pairing = load("4_link", "entity_linking.pairing")

__all__ = ["episodes", "chains", "encoding", "pairing"]
