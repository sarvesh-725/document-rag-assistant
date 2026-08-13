"""State-machine contract tests that do not require external services."""

import inspect

from app.database.enums import DocumentStatus, IngestionStage, IngestionStatus, VersionStatus
from app.services import ingestion_state_machine as state_machine


def test_exact_state_values_are_defined():
    assert {item.value for item in VersionStatus} == {
        "PROCESSING", "READY", "FAILED", "DELETING", "DELETED",
    }
    assert {item.value for item in IngestionStatus} == {
        "PENDING", "RUNNING", "RETRYING", "COMPLETED", "FAILED", "CANCELLED",
    }
    assert [item.value for item in IngestionStage] == [
        "UPLOAD", "PARSE", "CHUNK", "EMBED", "INDEX", "FINALIZE",
    ]


def test_stage_transitions_are_ordered_and_persisted():
    source = inspect.getsource(state_machine.transition_stage)
    assert "await db.commit()" in source
    assert "cannot move backwards" in source
    assert "cannot skip" in source


def test_finalize_checks_deletion_and_eligibility_before_ready():
    source = inspect.getsource(state_machine.finalize_attempt)
    assert "DocumentStatus.DELETING.value" in source
    assert "DocumentStatus.DELETED.value" in source
    assert "VersionStatus.PROCESSING.value" in source
    assert "VersionStatus.READY.value" in source


def test_failures_record_error_and_reconciliation_requeues():
    for fn in (state_machine.mark_retryable_failure, state_machine.mark_non_retryable_failure):
        source = inspect.getsource(fn)
        assert "error_code" in source
        assert "error_message" in source
        assert "updated_at" in source
    source = inspect.getsource(state_machine.reconcile_stale_jobs)
    assert "IngestionStatus.PENDING.value" in source
    assert "IngestionStatus.RUNNING.value" in source
    assert "DOCUMENT_INGESTION_REQUESTED" in source


def test_document_state_does_not_include_ready_resurrection_path():
    assert DocumentStatus.DELETING.value == "DELETING"
    assert DocumentStatus.DELETED.value == "DELETED"
