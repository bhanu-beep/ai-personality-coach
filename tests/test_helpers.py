from app import (
    build_changes_needed,
    build_persona,
    enrich_trait_scores_from_text,
    get_neuro_tip,
    get_tip,
    score_text_to_feature,
)


def test_score_text_default_when_empty():
    assert score_text_to_feature("") == 3


def test_score_text_positive_raises_and_negative_lowers():
    assert score_text_to_feature("I lead the team, plan and help calmly") > 3
    assert score_text_to_feature("I panic and avoid it, afraid and confused") < 3


def test_score_text_stays_within_1_to_5():
    assert 1 <= score_text_to_feature("panic " * 50) <= 5
    assert 1 <= score_text_to_feature("lead help plan " * 50) <= 5


def test_enrich_scores_keyword_and_stress_penalty():
    base = {t: 50 for t in ["conf", "disc", "lead", "neuro", "open", "agree", "extra"]}
    out = enrich_trait_scores_from_text("I am afraid and not sure", dict(base))
    assert out["neuro"] > base["neuro"]
    assert out["conf"] < base["conf"]


def test_tips_cover_score_bands():
    assert get_tip(90, "confidence").startswith("Excellent")
    assert get_tip(10, "confidence").startswith("Low")
    assert "stable" in get_neuro_tip(10).lower()


def test_persona_and_changes_needed_shape():
    scores = {"conf": 90, "disc": 80, "lead": 30, "neuro": 20, "open": 50, "agree": 60, "extra": 40}
    assert " with " in build_persona(scores)
    changes = build_changes_needed(scores)
    assert len(changes) == 3
    assert all("trait" in c and "how_to_change" in c for c in changes)
