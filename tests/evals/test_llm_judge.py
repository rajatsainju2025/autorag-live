from autorag_live.evals.llm_judge import (
    DeterministicJudge,
    OpenAIJudge,
    get_judge,
    load_judge_config,
    save_judge_config,
)


def test_deterministic_judge():
    judge = DeterministicJudge()

    # Test relevance
    relevance = judge.judge_answer_relevance(
        "What color is the sky?", "blue", "The sky is blue and beautiful."
    )
    assert 0.0 <= relevance <= 1.0

    # Test faithfulness
    faithfulness = judge.judge_faithfulness(
        "The sky is blue.", "The sky is blue and the grass is green."
    )
    assert 0.0 <= faithfulness <= 1.0


def test_openai_judge_fallback():
    # Test fallback when no API key
    judge = OpenAIJudge(api_key=None)

    relevance = judge.judge_answer_relevance("test", "answer", "context")
    assert 0.0 <= relevance <= 1.0

    faithfulness = judge.judge_faithfulness("answer", "context")
    assert 0.0 <= faithfulness <= 1.0


def test_get_judge():
    # Test factory function
    det_judge = get_judge("deterministic")
    assert isinstance(det_judge, DeterministicJudge)

    openai_judge = get_judge("openai", api_key=None)
    assert isinstance(openai_judge, OpenAIJudge)


def test_judge_config(tmp_path):
    judge = DeterministicJudge()
    config_path = tmp_path / "test_config.json"

    save_judge_config(judge, str(config_path))
    loaded = load_judge_config(str(config_path))

    assert loaded["type"] == "DeterministicJudge"
