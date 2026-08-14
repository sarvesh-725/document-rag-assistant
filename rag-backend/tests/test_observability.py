import json
import logging

from app.observability import MetricsRegistry, structured_log


def test_metrics_registry_records_counters_and_observations():
    registry = MetricsRegistry()
    registry.increment("queue_depth", 2)
    registry.observe("answer_latency_p50", 12.5)
    snapshot = registry.snapshot()
    assert snapshot["queue_depth"] == 2
    assert snapshot["answer_latency_p50_count"] == 1
    assert snapshot["answer_latency_p50_total"] == 12.5


def test_structured_log_excludes_sensitive_payloads(caplog):
    logger = logging.getLogger("observability-test")
    with caplog.at_level(logging.INFO, logger="observability-test"):
        structured_log(
            logger,
            "test_event",
            request_id="r1",
            user_id="u1",
            latency=4,
            status="COMPLETED",
            error_code=None,
            password="secret",
            content="private document text",
        )
    record = json.loads(caplog.records[-1].message)
    assert record["request_id"] == "r1"
    assert "password" not in record
    assert "content" not in record
