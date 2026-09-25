"""An empty or unknown question fails with its reason, not with a ZeroDivisionError.  python -m hunch.test_schema"""
from hunch.schema import Candidate, Question


def test_empty_question_raises_its_reason():
    for q, reason in [(Question("q", "weird", "?"), "unknown question type"),
                      (Question("q", "choice", "?"), "choice needs 2..255 candidates"),
                      (Question("q", "score", "?"), "score needs 2..10 levels")]:
        try:
            q.k
        except ValueError as e:
            assert reason in str(e), e
        else:
            raise AssertionError(f"{q.type}: no error")
    assert Question("q", "boolean", "?").k == 2
    assert Question("q", "choice", "?", candidates=[Candidate("a", "A"), Candidate("b", "B")]).k == 2


if __name__ == "__main__":
    test_empty_question_raises_its_reason(); print("schema ok")
