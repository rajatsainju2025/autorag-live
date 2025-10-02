import json
import os

from autorag_live.pipeline.acceptance_policy import (
    AcceptancePolicy,
    create_config_backup,
    safe_config_update,
)


def test_acceptance_policy_first_run(tmp_path):
    policy = AcceptancePolicy(best_runs_file=str(tmp_path / "best.json"))

    # No previous best - should accept
    result = policy.evaluate_change({}, runs_dir=str(tmp_path))
    assert result is True

    # Should have created best runs file
    assert os.path.exists(tmp_path / "best.json")


def test_acceptance_policy_improvement(tmp_path):
    best_file = tmp_path / "best.json"

    # Create fake previous best
    fake_best = {"metrics": {"f1": 0.5, "em": 0.3}, "run_id": "previous"}
    with open(best_file, "w") as f:
        json.dump(fake_best, f)

    policy = AcceptancePolicy(threshold=0.01, best_runs_file=str(best_file))

    # Current run will be better (small suite has ~1.0 F1)
    result = policy.evaluate_change({}, runs_dir=str(tmp_path))
    assert result is True


def test_config_backup(tmp_path):
    config_file = tmp_path / "test_config.json"
    config_file.write_text('{"test": "value"}')

    backup_path = create_config_backup(str(config_file))

    assert os.path.exists(backup_path)
    assert "backup_" in backup_path

    # Backup should have same content
    with open(backup_path) as f:
        backup_content = json.load(f)
    assert backup_content == {"test": "value"}


def test_safe_config_update(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"original": true}')

    def update_func():
        with open(config_file, "w") as f:
            json.dump({"updated": True}, f)

    policy = AcceptancePolicy(best_runs_file=str(tmp_path / "best.json"))

    # Should accept (first run)
    result = safe_config_update(update_func, [str(config_file)], policy)
    assert result is True

    # Config should be updated
    with open(config_file) as f:
        content = json.load(f)
    assert content == {"updated": True}
