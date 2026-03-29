from autorag_live.safety.guardrails import GroundingValidator, HallucinationDetector


def test_hallucination_detector_matches_claims_against_source_words():
    detector = HallucinationDetector()

    result = detector.detect(
        "Mars has two moons. Jupiter has a large storm.",
        ["Mars has two moons named Phobos and Deimos."],
        confidence_threshold=0.9,
    )

    assert result.is_safe is False
    assert any("claims lack source support" in issue for issue in result.issues)


def test_grounding_validator_scores_word_overlap():
    validator = GroundingValidator(threshold=0.6)

    grounded = validator.validate(
        "Mars has two moons.",
        ["Mars has two moons named Phobos and Deimos."],
    )
    ungrounded = validator.validate(
        "Saturn has one moon.",
        ["Mars has two moons named Phobos and Deimos."],
    )

    assert grounded.is_safe is True
    assert ungrounded.is_safe is False
