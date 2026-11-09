"""A chain's head may move between uploads without tripping the one-head index.

Needs a database with the migrations applied (``TEST_DATABASE_URL``); skips
otherwise. The store is two uploads of one chain whose head surface changes.
"""

import os
import uuid

import pytest

from src.inject import store as ST

DSN = os.environ.get("TEST_DATABASE_URL")
TENANT = "cccccccc-0000-4000-8000-00000000000c"


def make_store(head: str) -> dict:
    fid, eid, cid = (str(uuid.uuid5(uuid.NAMESPACE_DNS, k)) for k in ("f", "e", "c"))
    surfaces = ["Sarah Chen", "S. Chen"]
    return {
        "files": {fid: {"file_id": fid, "source": "upload", "external_id": "n.md",
                        "full_text_hash": "h", "render_version": 1,
                        "synced_at": "2026-09-19T00:00:00+00:00", "removed_at": None}},
        "episodes": {eid: {"episode_id": eid, "file_id": fid, "text": "Sarah Chen (S. Chen).",
                           "removed_at": None}},
        "chains": {cid: {"chain_id": cid, "episode_id": eid, "content_key": "k",
                         "head_surface": head, "entity_id": cid}},
        "candidates": {f"c{i}": {"candidate_id": f"c{i}", "episode_id": eid, "start": i,
                                 "end": i + 1, "text": s, "candidate_type": "proper_name",
                                 "origin": "parser", "chain_id": cid, "is_head": s == head}
                       for i, s in enumerate(surfaces)},
        "names": {},
    }


@pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")
def test_moving_the_head_between_uploads_does_not_violate_the_index() -> None:
    import psycopg

    with psycopg.connect(DSN) as conn:
        conn.execute("INSERT INTO tenants (id, name) VALUES (%s, 'head-move test')"
                     " ON CONFLICT (id) DO NOTHING", (TENANT,))
        conn.commit()
    ingest_dsn = DSN.replace("postgres:postgres@", "app_ingest_login:localdev-only@")
    try:
        first = make_store("Sarah Chen")
        ST.upload(first, dsn=ingest_dsn, tenant_id=TENANT, full=True)
        second = make_store("S. Chen")
        second["uploaded"] = first["uploaded"]
        second["uploaded"]["chains"] = {}  # force the chain (and its rows) into the delta
        ST.upload(second, dsn=ingest_dsn, tenant_id=TENANT)
        with psycopg.connect(DSN) as conn:
            rows = conn.execute("SELECT surface FROM chain_surfaces WHERE tenant_id = %s"
                                " AND is_head", (TENANT,)).fetchall()
        assert rows == [("S. Chen",)]
    finally:
        with psycopg.connect(DSN) as conn:  # surfaces has no cascade: clear in FK order
            for table in ("chain_surfaces", "chains", "entities", "surfaces", "tenants"):
                column = "id" if table == "tenants" else "tenant_id"
                conn.execute(f"DELETE FROM {table} WHERE {column} = %s", (TENANT,))
            conn.commit()
