"""EntityLookup finds likely entries in a list of known names with BM25."""

import random
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from collectra.resources.base import Resource
from collectra.resources.entity_lookup import EntityLookup, LookupHit

GENERA = [
    "Didymograptus", "Monograptus", "Climacograptus", "Tetragraptus", "Isograptus",
    "Dictyonema", "Pseudoclimacograptus", "Diplograptus", "Pristiograptus", "Orthograptus",
]  # fmt: skip


def lookup(tmp_path, entries=GENERA, **kwargs):
    path = tmp_path / "entities.txt"
    path.write_text("\n".join(entries) + "\n")
    return EntityLookup(
        "genera_lookup", file=path, description="Graptolite genera", **kwargs
    )


def names(hits):
    return [h.entity for h in hits]


# --- the model ----------------------------------------------------------------------


def test_resources_are_pydantic_models_with_a_name_and_description(tmp_path):
    item = lookup(tmp_path)
    assert isinstance(item, Resource) and isinstance(item, BaseModel)
    assert (item.name, item.description) == ("genera_lookup", "Graptolite genera")
    assert EntityLookup.model_fields["limit"].default == 10


def test_the_name_can_be_given_by_position(tmp_path):
    assert EntityLookup("x", file=lookup(tmp_path).file).name == "x"


def test_unknown_options_are_rejected_so_typos_show_up(tmp_path):
    with pytest.raises(ValidationError):
        EntityLookup("x", file=lookup(tmp_path).file, descripton="typo")


@pytest.mark.parametrize("bad", ["has space", "", "a/b", "x" * 65])
def test_the_name_must_be_usable_as_a_tool_name(tmp_path, bad):
    with pytest.raises(ValueError, match="name"):
        EntityLookup(bad, file=lookup(tmp_path).file)


def test_a_missing_file_is_reported_by_name_and_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="genera_lookup.*nowhere.txt"):
        EntityLookup("genera_lookup", file=tmp_path / "nowhere.txt")


def test_a_relative_file_is_read_from_where_the_lookup_was_created(
    tmp_path, monkeypatch
):
    (tmp_path / "list.txt").write_text("Alpha\nBeta\n")
    monkeypatch.chdir(tmp_path)
    item = EntityLookup("x", file="list.txt")
    monkeypatch.chdir("/")
    assert item.search("alpha")[0].entity == "Alpha"


def test_blank_lines_and_repeats_are_ignored(tmp_path):
    item = lookup(tmp_path, ["Alpha", "", "  ", "Alpha", " Beta "])
    assert item.entities == ["Alpha", "Beta"]


# --- searching ----------------------------------------------------------------------


def test_an_exact_entry_comes_first_and_is_marked(tmp_path):
    hits = lookup(tmp_path).search("Climacograptus")
    assert hits[0] == LookupHit(entity="Climacograptus", score=1.0, exact=True)
    assert all(not h.exact for h in hits[1:])


def test_matching_ignores_case_accents_and_punctuation(tmp_path):
    item = lookup(tmp_path, ["Café Münster, Vic.", "Other place"])
    assert item.search("cafe munster vic")[0].exact
    assert item.search("CAFÉ-MÜNSTER (vic)")[0].entity == "Café Münster, Vic."


def test_a_misspelling_still_finds_the_entry(tmp_path):
    item = lookup(tmp_path)
    assert item.search("Didymograpus")[0].entity == "Didymograptus"
    assert item.search("Climacogratus")[0].entity == "Climacograptus"
    assert item.search("Tetragaptus")[0].entity == "Tetragraptus"
    assert not item.search("Didymograpus")[0].exact


def test_a_part_of_a_longer_entry_finds_it(tmp_path):
    item = lookup(
        tmp_path, ["Parish of Axedale, allot 6", "Parish of Walwa", "Mount Wombat"]
    )
    assert item.search("Axedale")[0].entity == "Parish of Axedale, allot 6"
    assert item.search("wombat")[0].entity == "Mount Wombat"


def test_words_that_are_rare_count_for_more(tmp_path):
    item = lookup(
        tmp_path, ["Parish of A", "Parish of B", "Parish of C", "Parish of Axedale"]
    )
    assert item.search("parish axedale")[0].entity == "Parish of Axedale"


def test_scores_are_relative_to_the_best_hit(tmp_path):
    hits = lookup(tmp_path).search("graptus monograptus")
    assert hits[0].score == pytest.approx(1.0)
    assert all(0 < h.score <= 1 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_the_limit_caps_the_results_and_can_be_overridden(tmp_path):
    item = lookup(tmp_path, limit=3)
    assert len(item.search("graptus")) == 3
    assert len(item.search("graptus", limit=5)) == 5
    assert len(item.search("graptus", limit=100)) <= len(GENERA)


def test_nothing_in_common_gives_no_hits(tmp_path):
    assert lookup(tmp_path).search("zzzzqqq") == []
    assert lookup(tmp_path).search("   ") == []
    assert lookup(tmp_path).search("") == []


def test_results_are_deterministic_with_ties_in_file_order(tmp_path):
    item = lookup(tmp_path, ["Aaa one", "Bbb one", "Ccc one"])
    assert names(item.search("one")) == ["Aaa one", "Bbb one", "Ccc one"]
    assert item.search("one") == item.search("one")


def test_an_empty_list_finds_nothing(tmp_path):
    assert lookup(tmp_path, []).search("anything") == []


def test_one_character_slips_are_found_in_a_realistic_list(tmp_path):
    rng = random.Random(7)
    words = [
        "".join(
            rng.choice("abcdefghilmnoprstu") for _ in range(rng.randint(8, 14))
        ).title()
        for _ in range(300)
    ]
    item = lookup(tmp_path, words)
    misses = 0
    for word in words[:100]:
        i = rng.randrange(len(word))
        slip = rng.choice(["drop", "swap", "sub"])
        if slip == "drop":
            garbled = word[:i] + word[i + 1 :]
        elif slip == "swap" and i + 1 < len(word):
            garbled = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        else:
            garbled = word[:i] + "x" + word[i + 1 :]
        if word not in names(item.search(garbled, limit=3)):
            misses += 1
    assert misses <= 3


# --- as a tool for the model --------------------------------------------------------


def test_it_becomes_a_tool_named_after_the_resource(tmp_path):
    tool = lookup(tmp_path).as_tool()
    assert tool.name == "genera_lookup"
    assert "Graptolite genera" in tool.description
    assert "not" in tool.description and "complete" in tool.description
    assert list(tool.args) == ["query"]


def test_the_tool_lists_candidates_with_scores_and_exact_matches(tmp_path):
    text = lookup(tmp_path).as_tool().invoke({"query": "Monograptus"})
    assert text.splitlines()[0].startswith("Candidates for 'Monograptus'")
    assert "Monograptus" in text and "Monograptus" != text
    assert "Didymograptus" in text or "Pristiograptus" in text
    exact = lookup(tmp_path).as_tool().invoke({"query": "monograptus"})
    assert "exact match" in exact.splitlines()[1]


def test_the_tool_says_so_when_nothing_matches(tmp_path):
    text = lookup(tmp_path).as_tool().invoke({"query": "zzzzqqq"})
    assert "No candidates" in text and "not" in text and "exhaustive" in text


def test_the_tool_asks_for_a_query_when_it_is_empty(tmp_path):
    assert "query" in lookup(tmp_path).as_tool().invoke({"query": "  "}).lower()
