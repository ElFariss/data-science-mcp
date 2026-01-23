"""Version Manager for experiment tracking.

Handles automatic version numbering (v0.1, v0.2, etc.),
experiment folder creation, and metadata persistence.
"""

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


@dataclass
class ExperimentMetadata:
    """Metadata for a single experiment version."""
    version: str
    created_at: str
    modality: str = ""
    task_type: str = ""
    models_trained: list[str] = field(default_factory=list)
    best_model: str = ""
    best_score: float = 0.0
    metric_name: str = ""
    data_path: str = ""
    target_column: str = ""
    feature_config: str = ""
    training_time_seconds: float = 0.0
    parent_version: Optional[str] = None
    notes: str = ""


class VersionManager:
    """Manages experiment versions and metadata."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize version manager.
        
        Args:
            experiments_dir: Base directory for experiment storage
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.experiments_dir / "versions.json"
        self._versions: dict[str, ExperimentMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load version metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    data = json.load(f)
                    for version, meta in data.items():
                        self._versions[version] = ExperimentMetadata(**meta)
            except (json.JSONDecodeError, TypeError):
                self._versions = {}
    
    def _save_metadata(self) -> None:
        """Save version metadata to disk."""
        data = {v: asdict(m) for v, m in self._versions.items()}
        with open(self._metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_new_version(self, parent_version: Optional[str] = None) -> str:
        """
        Create a new experiment version.
        
        Args:
            parent_version: Optional parent version for lineage tracking
            
        Returns:
            New version identifier (e.g., "v0.3")
        """
        # Determine next version number
        if not self._versions:
            major, minor = 0, 1
        else:
            versions = sorted(self._versions.keys())
            latest = versions[-1]
            # Parse version like "v0.3" or "v1.2"
            parts = latest[1:].split('.')
            major, minor = int(parts[0]), int(parts[1])
            minor += 1
            if minor >= 10:
                major += 1
                minor = 0
        
        version = f"v{major}.{minor}"
        
        # Create experiment directory
        version_dir = self.experiments_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (version_dir / "models").mkdir(exist_ok=True)
        (version_dir / "plots").mkdir(exist_ok=True)
        (version_dir / "features").mkdir(exist_ok=True)
        
        # Create metadata
        self._versions[version] = ExperimentMetadata(
            version=version,
            created_at=datetime.now().isoformat(),
            parent_version=parent_version,
        )
        
        self._save_metadata()
        
        return version
    
    def get_latest_version(self) -> Optional[str]:
        """Get the most recent experiment version."""
        if not self._versions:
            return None
        return sorted(self._versions.keys())[-1]
    
    def get_version_path(self, version: str) -> Path:
        """Get the directory path for a version."""
        return self.experiments_dir / version
    
    def save_experiment_data(
        self,
        version: str,
        data: dict[str, Any],
        filename: str = "experiment_data.json"
    ) -> None:
        """
        Save experiment data to version folder.
        
        Args:
            version: Version identifier
            data: Data dictionary to save
            filename: Output filename
        """
        version_dir = self.get_version_path(version)
        with open(version_dir / filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_experiment_data(
        self,
        version: str,
        filename: str = "experiment_data.json"
    ) -> dict[str, Any]:
        """
        Load experiment data from version folder.
        
        Args:
            version: Version identifier
            filename: Input filename
            
        Returns:
            Loaded data dictionary
        """
        version_dir = self.get_version_path(version)
        filepath = version_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def update_metadata(self, version: str, **kwargs) -> None:
        """
        Update metadata for a version.
        
        Args:
            version: Version identifier
            **kwargs: Metadata fields to update
        """
        if version not in self._versions:
            raise ValueError(f"Version {version} not found")
        
        meta = self._versions[version]
        for key, value in kwargs.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
        
        self._save_metadata()
    
    def get_metadata(self, version: str) -> Optional[ExperimentMetadata]:
        """Get metadata for a version."""
        return self._versions.get(version)
    
    def list_all_versions(self) -> list[str]:
        """List all experiment versions."""
        return sorted(self._versions.keys())
    
    def get_version_history(self) -> list[dict]:
        """Get version history with key metrics."""
        history = []
        for version in self.list_all_versions():
            meta = self._versions[version]
            history.append({
                "version": version,
                "created_at": meta.created_at,
                "best_model": meta.best_model,
                "best_score": meta.best_score,
                "metric_name": meta.metric_name,
            })
        return history
    
    def copy_to_version(self, source_path: str, version: str, dest_name: str) -> str:
        """
        Copy a file to a version's directory.
        
        Args:
            source_path: Path to source file
            version: Target version
            dest_name: Destination filename
            
        Returns:
            Path to copied file
        """
        version_dir = self.get_version_path(version)
        dest_path = version_dir / dest_name
        shutil.copy(source_path, dest_path)
        return str(dest_path)
    
    def save_model(self, version: str, model: Any, model_name: str) -> str:
        """
        Save a trained model to version folder.
        
        Args:
            version: Version identifier
            model: Model object to save
            model_name: Name for the model file
            
        Returns:
            Path to saved model
        """
        import joblib
        
        models_dir = self.get_version_path(version) / "models"
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        
        return str(model_path)
    
    def load_model(self, version: str, model_name: str) -> Any:
        """
        Load a model from version folder.
        
        Args:
            version: Version identifier
            model_name: Name of the model file
            
        Returns:
            Loaded model object
        """
        import joblib
        
        models_dir = self.get_version_path(version) / "models"
        model_path = models_dir / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found in {version}")
        
        return joblib.load(model_path)
