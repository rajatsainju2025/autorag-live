from typing import Dict, Any, Optional
import json
import os
import shutil
from datetime import datetime

from autorag_live.evals.small import run_small_suite


class AcceptancePolicy:
    """
    Policy for accepting or reverting model/config changes based on evaluation metrics.
    """
    
    def __init__(self, 
                 threshold: float = 0.01, 
                 metric_key: str = "f1",
                 best_runs_file: str = "best_runs.json"):
        self.threshold = threshold
        self.metric_key = metric_key
        self.best_runs_file = best_runs_file
        
    def get_current_best(self) -> Optional[Dict[str, Any]]:
        """Get the current best run metrics."""
        if not os.path.exists(self.best_runs_file):
            return None
            
        with open(self.best_runs_file, 'r') as f:
            best_data = json.load(f)
        return best_data
        
    def evaluate_change(self, 
                       config_backup_paths: Dict[str, str],
                       runs_dir: str = "runs") -> bool:
        """
        Evaluate if recent changes should be accepted or reverted.
        
        Args:
            config_backup_paths: Dict mapping config files to their backup paths
            runs_dir: Directory containing evaluation runs
            
        Returns:
            True if changes should be accepted, False if reverted
        """
        # Run evaluation
        current_run = run_small_suite(runs_dir)
        current_metric = current_run["metrics"][self.metric_key]
        
        # Get baseline
        best_run = self.get_current_best()
        if best_run is None:
            # First run - accept and save as best
            self._update_best(current_run)
            return True
            
        best_metric = best_run["metrics"][self.metric_key]
        improvement = current_metric - best_metric
        
        if improvement >= self.threshold:
            # Accept - update best
            self._update_best(current_run)
            self._cleanup_backups(config_backup_paths)
            print(f"✅ Changes ACCEPTED: {self.metric_key} improved by {improvement:.4f}")
            return True
        else:
            # Revert - restore backups
            self._revert_configs(config_backup_paths)
            print(f"❌ Changes REVERTED: {self.metric_key} change {improvement:.4f} < threshold {self.threshold}")
            return False
            
    def _update_best(self, run_data: Dict[str, Any]) -> None:
        """Update the best run record."""
        with open(self.best_runs_file, 'w') as f:
            json.dump(run_data, f, indent=2)
            
    def _cleanup_backups(self, backup_paths: Dict[str, str]) -> None:
        """Remove backup files after successful acceptance."""
        for backup_path in backup_paths.values():
            if os.path.exists(backup_path):
                os.remove(backup_path)
                
    def _revert_configs(self, backup_paths: Dict[str, str]) -> None:
        """Restore config files from backups."""
        for original_path, backup_path in backup_paths.items():
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, original_path)
                print(f"Reverted {original_path} from {backup_path}")


def create_config_backup(file_path: str) -> str:
    """Create a timestamped backup of a config file."""
    if not os.path.exists(file_path):
        return ""
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    return backup_path


def safe_config_update(update_func, config_files: list, policy: AcceptancePolicy):
    """
    Safely update configuration files with automatic revert on failure.
    
    Args:
        update_func: Function that performs the config updates
        config_files: List of config file paths that will be modified
        policy: AcceptancePolicy instance for evaluation
    """
    # Create backups
    backups = {}
    for config_file in config_files:
        backup_path = create_config_backup(config_file)
        if backup_path:
            backups[config_file] = backup_path
    
    try:
        # Apply updates
        update_func()
        
        # Evaluate and potentially revert
        accepted = policy.evaluate_change(backups)
        return accepted
        
    except Exception as e:
        # Revert on error
        policy._revert_configs(backups)
        print(f"Error during update, reverted: {e}")
        return False
