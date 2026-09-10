from ddsr_bench.grading.categorical import categorical


def test_multiple_choice_labels() -> None:
    assert categorical("A", "A")
    assert categorical(" a ", "A")
    assert not categorical("B", "A")


def test_rejects_non_labels() -> None:
    assert not categorical("", "")
    assert not categorical(1, 1)
    assert not categorical(["A"], ["A"])
