from app.evaluation.dataset_generation import SourceContext, _validated_case
from app.evaluation.datasets import EvaluationSample


def _contexts():
    return [
        SourceContext("doc-a", "a.txt", "alpha", 1, 1, None),
        SourceContext("doc-b", "b.txt", "beta answer", 2, 2, None),
    ]


def test_generated_answerable_case_keeps_all_documents_in_retrieval_scope():
    case = _validated_case(
        {
            "category": "direct_lookup",
            "question": "What is beta?",
            "ground_truth": "beta answer",
            "sources": [{"document_id": "doc-b", "page": 2}],
        },
        _contexts(),
        1,
    )

    assert case["selected_documents"] == ["doc-a", "doc-b"]
    assert case["expected_sources"] == [{"document_id": "doc-b", "page": 2}]


def test_generated_no_answer_case_keeps_scope_and_has_no_ground_truth_source():
    case = _validated_case(
        {
            "category": "no_answer",
            "question": "Who won an unrelated event?",
            "ground_truth": "",
            "sources": [],
        },
        _contexts(),
        1,
    )

    assert case["selected_documents"] == ["doc-a", "doc-b"]
    assert case["expected_sources"] == []
    assert case["ground_truth"] == ""


def test_dataset_loader_does_not_derive_retrieval_scope_from_ground_truth():
    sample = EvaluationSample.from_dict(
        {
            "question": "What is beta?",
            "ground_truth": "beta answer",
            "expected_sources": [{"document_id": "doc-b", "page": 2}],
        }
    )

    assert sample.expected_document_ids == ["doc-b"]
    assert sample.selected_document_ids == []
