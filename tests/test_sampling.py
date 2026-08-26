from src.util.sampling import sample_per_difficulty


def _difficulty(record: dict) -> str | None:
    return record["difficulty"]


def test_missing_limit_preserves_all_records_and_order():
    records = [
        {"id": 1, "difficulty": "medium"},
        {"id": 2, "difficulty": None},
        {"id": 3, "difficulty": "easy"},
    ]

    assert sample_per_difficulty(records, None, _difficulty) == records


def test_limit_samples_each_difficulty_deterministically():
    records = [
        {"id": f"{difficulty}-{index}", "difficulty": difficulty}
        for difficulty in ("easy", "medium", "hard", "extra")
        for index in range(20)
    ]

    first = sample_per_difficulty(records, 5, _difficulty)
    second = sample_per_difficulty(records, 5, _difficulty)

    assert first == second
    for difficulty in ("easy", "medium", "hard", "extra"):
        assert sum(record["difficulty"] == difficulty for record in first) == 5


def test_each_difficulty_sample_is_independent():
    records = [
        {"id": f"{difficulty}-{index}", "difficulty": difficulty}
        for difficulty in ("easy", "hard")
        for index in range(20)
    ]

    combined = sample_per_difficulty(records, 5, _difficulty)
    hard_only = sample_per_difficulty(
        [record for record in records if record["difficulty"] == "hard"],
        5,
        _difficulty,
    )

    assert [record for record in combined if record["difficulty"] == "hard"] == hard_only


def test_limit_is_capped_by_available_records():
    records = [{"id": 1, "difficulty": "easy"}]

    assert sample_per_difficulty(records, 100, _difficulty) == records


def test_negative_limit_is_rejected():
    try:
        sample_per_difficulty([], -1, _difficulty)
    except ValueError as error:
        assert str(error) == "limit must be greater than or equal to zero"
    else:
        raise AssertionError("negative limit should be rejected")
