"""Run: uv run dotenv run -- python debug_search.py """
import asyncio
from dotenv import load_dotenv
load_dotenv()

GROUP_ID = "testing-1"

async def main() -> None:
    from services.neo4j import Neo4jGraphDriver
    driver = Neo4jGraphDriver.from_env()

    # 1. Pick a known edge (check books ↔ bog gap) and inspect it fully
    print("=== Raw edge inspection: check books ↔ bog gap ===")
    rows = await driver.execute_query("""
        MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
        WHERE n.name = 'check books' AND n.group_id = $gid
        RETURN
            n.uuid AS n_uuid,
            id(n) AS n_internal_id,
            m.uuid AS m_uuid,
            m.name AS m_name,
            e.source_node_uuid AS e_src,
            e.target_node_uuid AS e_tgt,
            e.fact AS fact,
            startNode(e).uuid AS actual_start_uuid,
            endNode(e).uuid AS actual_end_uuid
        LIMIT 3
    """, gid=GROUP_ID)
    for r in rows:
        print(f"  n(check books).uuid   = {r['n_uuid']}")
        print(f"  m({r['m_name']}).uuid  = {r['m_uuid']}")
        print(f"  edge.source_node_uuid = {r['e_src']}")
        print(f"  edge.target_node_uuid = {r['e_tgt']}")
        print(f"  startNode(e).uuid     = {r['actual_start_uuid']}  ← actual graph direction start")
        print(f"  endNode(e).uuid       = {r['actual_end_uuid']}  ← actual graph direction end")
        print(f"  n_uuid == e_src?      {r['n_uuid'] == r['e_src']}")
        print(f"  fact: {str(r['fact'])[:80]}")
        print()

    # 2. Does e.source_node_uuid match startNode(e).uuid across all edges?
    print("=== UUID alignment across all edges ===")
    mismatch = await driver.execute_query("""
        MATCH (n:Entity {group_id: $gid})-[e:RELATES_TO]->(m:Entity {group_id: $gid})
        WHERE e.source_node_uuid <> n.uuid OR e.target_node_uuid <> m.uuid
        RETURN count(e) AS mismatched
    """, gid=GROUP_ID)
    total = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {group_id: $gid}]->() RETURN count(e) AS n", gid=GROUP_ID
    )
    print(f"  total directed edges:          {total[0]['n']}")
    print(f"  edges with UUID mismatch:      {mismatch[0]['mismatched']}")

    await driver.close()

asyncio.run(main())
