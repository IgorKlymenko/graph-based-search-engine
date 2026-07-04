from __future__ import annotations

import asyncio
import logging
import os
from typing import NamedTuple

from retrieval.config import PPRSearchConfig
from retrieval.exceptions import GraphLoadError, GraphTooLargeError
from retrieval.interfaces import GraphDriver

log = logging.getLogger(__name__)

_LOAD_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("RETRIEVAL_PPR_CONCURRENCY", "4")))

_NODE_QUERY = (
    "MATCH (n:Entity {group_id: $group_id}) "
    "RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary"
)

_EDGE_QUERY = (
    "MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO]->(m:Entity {group_id: $group_id}) "
    "RETURN e.source_node_uuid AS source_uuid, e.target_node_uuid AS target_uuid"
)


class GraphNode(NamedTuple):
    uuid: str
    name: str
    summary: str


class GraphEdge(NamedTuple):
    source_uuid: str
    target_uuid: str


async def load_group(
    driver: GraphDriver,
    group_id: str,
    config: PPRSearchConfig,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Load all nodes and structural edges for a group."""
    async with _LOAD_SEMAPHORE:
        try:
            node_rows = await driver.execute_query(_NODE_QUERY, group_id=group_id)
        except Exception as exc:
            raise GraphLoadError("Failed to load graph nodes") from exc

        try:
            edge_rows = await driver.execute_query(_EDGE_QUERY, group_id=group_id)
        except Exception as exc:
            raise GraphLoadError("Failed to load graph edges") from exc

    nodes = [
        GraphNode(uuid=r["uuid"], name=r["name"], summary=r.get("summary") or "")
        for r in node_rows
    ]
    edges = [
        GraphEdge(source_uuid=r["source_uuid"], target_uuid=r["target_uuid"])
        for r in edge_rows
    ]

    if len(nodes) > config.max_nodes or len(edges) > config.max_edges:
        raise GraphTooLargeError(
            num_nodes=len(nodes),
            num_edges=len(edges),
            limits={"max_nodes": config.max_nodes, "max_edges": config.max_edges},
        )

    return nodes, edges
