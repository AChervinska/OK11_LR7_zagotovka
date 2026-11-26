"""Тести app/retrain_policy.py — крок 12 (поглиблений рівень)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.advanced

from app.retrain_policy import RetrainTriggers, should_retrain


def test_no_triggers_no_retrain():
    ok, reason = should_retrain(RetrainTriggers())
    assert ok is False
    assert "жоден" in reason


def test_scheduled_trigger():
    ok, reason = should_retrain(RetrainTriggers(scheduled_due=True))
    assert ok is True
    assert "розклад" in reason


def test_persistent_drift_trigger():
    ok, reason = should_retrain(RetrainTriggers(drift_persistent=True))
    assert ok is True
    assert "дрейф" in reason


def test_quality_drop_below_threshold_does_not_trigger():
    ok, _ = should_retrain(RetrainTriggers(quality_drop=0.02, min_quality_drop=0.05))
    assert ok is False


def test_quality_drop_above_threshold_triggers():
    ok, reason = should_retrain(RetrainTriggers(quality_drop=0.09, min_quality_drop=0.05))
    assert ok is True
    assert "якост" in reason


def test_quality_drop_reason_wins_over_schedule():
    ok, reason = should_retrain(
        RetrainTriggers(scheduled_due=True, drift_persistent=True, quality_drop=0.2)
    )
    assert ok is True
    assert "якост" in reason  # найтерміновіша причина в поясненні
