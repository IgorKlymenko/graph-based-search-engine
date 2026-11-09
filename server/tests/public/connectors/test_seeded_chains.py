"""Stage 2 folds connector seeds into the model's chains instead of beside them."""

from src.inject import chains as CH
from src.inject.utils import spans as SP

TEXT = "function ingest (src.inject.main.ingest)\nThe ingest function loads it."


def at(needle: str, nth: int = 0) -> tuple[int, int]:
    start = -1
    for _ in range(nth + 1):
        start = TEXT.index(needle, start + 1)
    return start, start + len(needle)


def span(sid, where, ctype=SP.NOMINAL, origin="parser"):
    start, end = where
    return SP.Span(sid, start, end, TEXT[start:end], ctype, origin)


def episode(seeds):
    return {"episode_id": "e1", "text": TEXT, "seeds": seeds}


def test_seeds_become_proper_name_spans_with_connector_origin() -> None:
    (chain,) = CH.seeded_chains(episode([
        {"key": "src.inject.main.ingest",
         "spans": [[9, 15, "ingest"], [17, 39, "src.inject.main.ingest"]]}]))
    assert [(s.text, s.candidate_type, s.origin) for s in chain] == [
        ("ingest", SP.PROPER_NAME, CH.SEED_ORIGIN),
        ("src.inject.main.ingest", SP.PROPER_NAME, CH.SEED_ORIGIN)]


def test_a_seed_span_that_does_not_match_the_text_is_dropped(capsys) -> None:
    chains = CH.seeded_chains(episode([{"key": "k", "spans": [[0, 8, "nonsense"]]}]))
    assert chains == []
    assert "seed spans did not match" in capsys.readouterr().out


def test_a_seed_absorbs_the_model_chain_that_names_the_same_thing() -> None:
    """Two chains holding "ingest" in one episode would read as a contested name."""
    model = [[span("c1", at("ingest", 2)), span("c2", at("it"), SP.PRONOUN)],
             [span("c3", at("function"))]]
    seed = [[span("s0", at("ingest"), SP.PROPER_NAME, "connector"),
             span("s1", at("src.inject.main.ingest"), SP.PROPER_NAME, "connector")]]
    merged = CH.merge_seeded(model, seed)
    assert len(merged) == 2
    absorbed = next(c for c in merged if any(s.origin == "connector" for s in c))
    assert sorted(s.text for s in absorbed) == [
        "ingest", "ingest", "it", "src.inject.main.ingest"]
    assert absorbed[0].candidate_type == SP.PROPER_NAME  # head is a name


def test_on_identical_offsets_the_seed_span_wins() -> None:
    model = [[span("c1", at("ingest"))]]  # the parser saw "ingest" as a nominal
    seed = [[span("s0", at("ingest"), SP.PROPER_NAME, "connector")]]
    (chain,) = CH.merge_seeded(model, seed)
    assert len(chain) == 1
    assert chain[0].origin == "connector"


def test_untouched_model_chains_pass_through_unchanged() -> None:
    model = [[span("c3", at("function"))]]
    seed = [[span("s0", at("ingest"), SP.PROPER_NAME, "connector")]]
    merged = CH.merge_seeded(model, seed)
    assert model[0] in merged and len(merged) == 2


def test_no_seeds_is_the_old_behaviour() -> None:
    model = [[span("c3", at("function"))]]
    assert CH.merge_seeded(model, []) is model


def test_process_episode_stores_the_seed_as_a_chain(monkeypatch) -> None:
    store = {"episodes": {"e1": episode([
        {"key": "src.inject.main.ingest",
         "spans": [[9, 15, "ingest"], [17, 39, "src.inject.main.ingest"]]}])}}
    monkeypatch.setattr(CH, "_complete_json",
                        lambda **kw: {"new_spans": [], "chains": []})
    candidates, chains, failure = CH.process_episode(store, "e1", model="stub")
    assert failure is None
    (chain,) = chains
    assert chain["head_surface"] == "src.inject.main.ingest"
    assert {c["text"] for c in candidates} == {"ingest", "src.inject.main.ingest"}
    assert all(c["origin"] == CH.SEED_ORIGIN for c in candidates)
