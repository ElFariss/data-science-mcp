"""Experiment Tracker for logging all training runs.

Provides a unified interface for tracking metrics, parameters,
artifacts, and comparing experiments.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


@dataclass
class ExperimentRun:
    """A single experiment run."""
    run_id: str
    version: str
    model_name: str
    started_at: str
    finished_at: Optional[str] = None
    status: str = "running"  # running, completed, failed
    parameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    error_message: Optional[str] = None
    training_time_seconds: float = 0.0


class ExperimentTracker:
    """Tracks all experiment runs and provides comparison utilities."""
    
    def __init__(self, tracking_dir: str = "experiments/tracking"):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_dir: Directory for tracking data
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self._runs_file = self.tracking_dir / "runs.json"
        self._runs: dict[str, ExperimentRun] = {}
        self._load_runs()
        self._run_counter = len(self._runs)
    
    def _load_runs(self) -> None:
        """Load runs from disk."""
        if self._runs_file.exists():
            try:
                with open(self._runs_file, 'r') as f:
                    data = json.load(f)
                    for run_id, run_data in data.items():
                        self._runs[run_id] = ExperimentRun(**run_data)
            except (json.JSONDecodeError, TypeError):
                self._runs = {}
    
    def _save_runs(self) -> None:
        """Save runs to disk."""
        data = {run_id: asdict(run) for run_id, run in self._runs.items()}
        with open(self._runs_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def start_run(
        self,
        version: str,
        model_name: str,
        parameters: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """
        Start a new experiment run.
        
        Args:
            version: Experiment version
            model_name: Name of the model being trained
            parameters: Model parameters/hyperparameters
            tags: Optional tags for categorization
            
        Returns:
            Run ID
        """
        self._run_counter += 1
        run_id = f"run_{self._run_counter:04d}"
        
        self._runs[run_id] = ExperimentRun(
            run_id=run_id,
            version=version,
            model_name=model_name,
            started_at=datetime.now().isoformat(),
            parameters=parameters or {},
            tags=tags or [],
        )
        
        self._save_runs()
        return run_id
    
    def log_metric(self, run_id: str, name: str, value: float) -> None:
        """
        Log a metric for a run.
        
        Args:
            run_id: Run identifier
            name: Metric name
            value: Metric value
        """
        if run_id in self._runs:
            self._runs[run_id].metrics[name] = value
            self._save_runs()
    
    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """
        Log multiple metrics for a run.
        
        Args:
            run_id: Run identifier
            metrics: Dictionary of metric names to values
        """
        if run_id in self._runs:
            self._runs[run_id].metrics.update(metrics)
            self._save_runs()
    
    def log_parameter(self, run_id: str, name: str, value: Any) -> None:
        """
        Log a parameter for a run.
        
        Args:
            run_id: Run identifier
            name: Parameter name
            value: Parameter value
        """
        if run_id in self._runs:
            self._runs[run_id].parameters[name] = value
            self._save_runs()
    
    def log_artifact(self, run_id: str, artifact_path: str) -> None:
        """
        Log an artifact path for a run.
        
        Args:
            run_id: Run identifier
            artifact_path: Path to the artifact
        """
        if run_id in self._runs:
            self._runs[run_id].artifacts.append(artifact_path)
            self._save_runs()
    
    def end_run(
        self,
        run_id: str,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> None:
        """
        End an experiment run.
        
        Args:
            run_id: Run identifier
            status: Final status (completed, failed)
            error_message: Optional error message if failed
        """
        if run_id in self._runs:
            run = self._runs[run_id]
            run.finished_at = datetime.now().isoformat()
            run.status = status
            run.error_message = error_message
            
            # Calculate training time
            if run.started_at:
                start = datetime.fromisoformat(run.started_at)
                end = datetime.fromisoformat(run.finished_at)
                run.training_time_seconds = (end - start).total_seconds()
            
            self._save_runs()
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run."""
        return self._runs.get(run_id)
    
    def get_runs_by_version(self, version: str) -> list[ExperimentRun]:
        """Get all runs for a version."""
        return [run for run in self._runs.values() if run.version == version]
    
    def get_runs_by_model(self, model_name: str) -> list[ExperimentRun]:
        """Get all runs for a specific model."""
        return [run for run in self._runs.values() if run.model_name == model_name]
    
    def get_best_run(
        self,
        metric_name: str,
        higher_is_better: bool = True,
        version: Optional[str] = None,
    ) -> Optional[ExperimentRun]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to compare
            higher_is_better: Whether higher values are better
            version: Optional version filter
            
        Returns:
            Best run or None
        """
        runs = self._runs.values()
        
        if version:
            runs = [r for r in runs if r.version == version]
        
        runs_with_metric = [
            r for r in runs 
            if metric_name in r.metrics and r.status == "completed"
        ]
        
        if not runs_with_metric:
            return None
        
        if higher_is_better:
            return max(runs_with_metric, key=lambda r: r.metrics[metric_name])
        else:
            return min(runs_with_metric, key=lambda r: r.metrics[metric_name])
    
    def compare_runs(
        self,
        run_ids: list[str],
        metrics: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Optional list of metrics to include
            
        Returns:
            List of run summaries for comparison
        """
        comparisons = []
        
        for run_id in run_ids:
            run = self._runs.get(run_id)
            if run:
                summary = {
                    "run_id": run.run_id,
                    "model_name": run.model_name,
                    "version": run.version,
                    "status": run.status,
                    "training_time": run.training_time_seconds,
                }
                
                if metrics:
                    for m in metrics:
                        summary[m] = run.metrics.get(m)
                else:
                    summary["metrics"] = run.metrics
                
                comparisons.append(summary)
        
        return comparisons
    
    def get_experiment_summary(self) -> dict:
        """Get a summary of all experiments."""
        total_runs = len(self._runs)
        completed = sum(1 for r in self._runs.values() if r.status == "completed")
        failed = sum(1 for r in self._runs.values() if r.status == "failed")
        
        versions = set(r.version for r in self._runs.values())
        models = set(r.model_name for r in self._runs.values())
        
        return {
            "total_runs": total_runs,
            "completed": completed,
            "failed": failed,
            "running": total_runs - completed - failed,
            "unique_versions": len(versions),
            "unique_models": len(models),
            "versions": sorted(versions),
            "models": sorted(models),
        }
