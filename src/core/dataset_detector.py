"""Dataset Detector for automatic modality and task detection.

Detects whether data is tabular, time series, vision, or NLP,
and identifies the task type (classification, regression, etc.).
"""

import os
from pathlib import Path
from typing import Optional, Any
import warnings


class DatasetDetector:
    """Detects dataset modality, task type, and data characteristics."""
    
    # File extensions for different modalities
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    TEXT_EXTENSIONS = {'.txt', '.json', '.jsonl'}
    TABULAR_EXTENSIONS = {'.csv', '.tsv', '.parquet', '.feather', '.xlsx', '.xls'}
    
    # Columns that suggest time series data
    TIME_COLUMNS = {
        'date', 'datetime', 'timestamp', 'time', 'ds', 'dt',
        'year', 'month', 'day', 'hour', 'minute', 'second',
        'created_at', 'updated_at', 'created', 'updated',
    }
    
    # Columns that suggest NLP data
    TEXT_COLUMNS = {
        'text', 'content', 'description', 'comment', 'review',
        'title', 'body', 'message', 'sentence', 'document',
        'question', 'answer', 'query', 'passage',
    }
    
    def detect_modality(self, data_path: str) -> str:
        """
        Detect the modality of the dataset.
        
        Args:
            data_path: Path to the dataset file or directory
            
        Returns:
            Modality string: 'tabular', 'timeseries', 'vision', or 'nlp'
        """
        path = Path(data_path)
        
        # If it's a directory, check for images
        if path.is_dir():
            files = list(path.rglob('*'))
            image_count = sum(1 for f in files if f.suffix.lower() in self.IMAGE_EXTENSIONS)
            if image_count > 0:
                return 'vision'
            
            # Check for text files
            text_count = sum(1 for f in files if f.suffix.lower() in self.TEXT_EXTENSIONS)
            if text_count > len(files) * 0.5:
                return 'nlp'
            
            # Default to tabular
            return 'tabular'
        
        # Single file
        suffix = path.suffix.lower()
        
        if suffix in self.IMAGE_EXTENSIONS:
            return 'vision'
        
        if suffix in self.TABULAR_EXTENSIONS:
            # Need to analyze content to distinguish tabular vs timeseries vs nlp
            return self._detect_from_tabular_file(path)
        
        if suffix in self.TEXT_EXTENSIONS:
            return 'nlp'
        
        # Default
        return 'tabular'
    
    def _detect_from_tabular_file(self, path: Path) -> str:
        """Detect modality from a tabular file by analyzing columns."""
        import pandas as pd
        
        try:
            # Read just the header and a few rows
            if path.suffix == '.parquet':
                df = pd.read_parquet(path).head(100)
            elif path.suffix in {'.xlsx', '.xls'}:
                df = pd.read_excel(path, nrows=100)
            else:
                df = pd.read_csv(path, nrows=100)
            
            columns_lower = {c.lower() for c in df.columns}
            
            # Check for time columns
            time_cols = columns_lower & self.TIME_COLUMNS
            if time_cols:
                # Further check if it looks like time series
                if len(df.columns) <= 10 and len(time_cols) >= 1:
                    return 'timeseries'
            
            # Check for text columns
            text_cols = columns_lower & self.TEXT_COLUMNS
            if text_cols:
                # Check if text columns have substantial text
                for col in df.columns:
                    if col.lower() in self.TEXT_COLUMNS:
                        if df[col].dtype == 'object':
                            avg_len = df[col].astype(str).str.len().mean()
                            if avg_len > 100:  # Substantial text
                                return 'nlp'
            
            return 'tabular'
            
        except Exception:
            return 'tabular'
    
    def detect_task_type(
        self,
        data_path: str,
        target_column: Optional[str] = None
    ) -> str:
        """
        Detect the task type from the target column.
        
        Args:
            data_path: Path to the dataset
            target_column: Target column name
            
        Returns:
            Task type: 'classification', 'regression', 'multiclass', 
            'multilabel', 'forecasting', etc.
        """
        import pandas as pd
        
        path = Path(data_path)
        
        if not path.is_file():
            return 'unknown'
        
        try:
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            elif path.suffix in {'.xlsx', '.xls'}:
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            
            # Try to find target column
            if target_column is None:
                target_column = self._guess_target_column(df)
            
            if target_column is None or target_column not in df.columns:
                return 'unknown'
            
            target = df[target_column]
            
            # Check for numeric vs categorical
            if pd.api.types.is_numeric_dtype(target):
                n_unique = target.nunique()
                n_total = len(target)
                
                # If few unique values relative to total, likely classification
                if n_unique <= 10:
                    if n_unique == 2:
                        return 'binary_classification'
                    return 'multiclass_classification'
                elif n_unique / n_total < 0.05:
                    return 'multiclass_classification'
                else:
                    return 'regression'
            else:
                # Categorical target
                n_unique = target.nunique()
                if n_unique == 2:
                    return 'binary_classification'
                return 'multiclass_classification'
                
        except Exception:
            return 'unknown'
    
    def _guess_target_column(self, df) -> Optional[str]:
        """Attempt to guess the target column."""
        common_targets = [
            'target', 'label', 'y', 'class', 'outcome', 'result',
            'is_fraud', 'is_spam', 'is_positive', 'is_negative',
            'survived', 'price', 'revenue', 'sales', 'rating',
        ]
        
        for col in common_targets:
            if col in df.columns:
                return col
            if col.lower() in [c.lower() for c in df.columns]:
                for c in df.columns:
                    if c.lower() == col.lower():
                        return c
        
        # Last column is often target
        return df.columns[-1]
    
    def analyze_dataset(
        self,
        data_path: str,
        target_column: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Analyze dataset and return comprehensive info.
        
        Args:
            data_path: Path to the dataset
            target_column: Optional target column name
            
        Returns:
            Dictionary with dataset information
        """
        import pandas as pd
        
        modality = self.detect_modality(data_path)
        
        if modality == 'vision':
            return self._analyze_vision_dataset(data_path)
        
        path = Path(data_path)
        
        try:
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            elif path.suffix in {'.xlsx', '.xls'}:
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            
            if target_column is None:
                target_column = self._guess_target_column(df)
            
            task_type = self.detect_task_type(data_path, target_column)
            
            # Column analysis
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            return {
                "modality": modality,
                "task_type": task_type,
                "n_samples": len(df),
                "n_features": len(df.columns) - (1 if target_column else 0),
                "target_column": target_column,
                "numeric_features": len(numeric_cols),
                "categorical_features": len(categorical_cols),
                "datetime_features": len(datetime_cols),
                "column_names": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
            }
            
        except Exception as e:
            return {
                "modality": modality,
                "task_type": "unknown",
                "error": str(e),
            }
    
    def _analyze_vision_dataset(self, data_path: str) -> dict[str, Any]:
        """Analyze a vision dataset (directory of images)."""
        path = Path(data_path)
        
        if path.is_file():
            return {
                "modality": "vision",
                "task_type": "single_image",
                "n_samples": 1,
            }
        
        # Collect image info
        images = list(path.rglob('*'))
        images = [f for f in images if f.suffix.lower() in self.IMAGE_EXTENSIONS]
        
        # Check for class directories
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            class_counts = {}
            for subdir in subdirs:
                class_images = [f for f in subdir.rglob('*') 
                               if f.suffix.lower() in self.IMAGE_EXTENSIONS]
                class_counts[subdir.name] = len(class_images)
            
            return {
                "modality": "vision",
                "task_type": "image_classification",
                "n_samples": len(images),
                "n_classes": len(class_counts),
                "class_distribution": class_counts,
            }
        
        return {
            "modality": "vision",
            "task_type": "unknown",
            "n_samples": len(images),
        }
    
    def check_data_quality(self, data_path: str) -> dict[str, Any]:
        """
        Check data quality issues.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Dictionary with quality metrics
        """
        import pandas as pd
        
        path = Path(data_path)
        
        if not path.is_file() or path.suffix.lower() not in self.TABULAR_EXTENSIONS:
            return {"applicable": False}
        
        try:
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            
            n_samples = len(df)
            
            # Missing values
            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / n_samples * 100).round(2)
            cols_with_missing = missing_pct[missing_pct > 0].to_dict()
            
            # Duplicates
            n_duplicates = df.duplicated().sum()
            
            # Constant features
            constant_features = [
                col for col in df.columns 
                if df[col].nunique() <= 1
            ]
            
            # High cardinality categoricals
            high_cardinality = []
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() > n_samples * 0.5:
                    high_cardinality.append(col)
            
            # Potential identifier columns
            potential_ids = []
            for col in df.columns:
                if df[col].nunique() == n_samples:
                    potential_ids.append(col)
            
            return {
                "applicable": True,
                "missing_percentage": round(df.isnull().sum().sum() / df.size * 100, 2),
                "columns_with_missing": cols_with_missing,
                "duplicates": int(n_duplicates),
                "duplicate_percentage": round(n_duplicates / n_samples * 100, 2),
                "constant_features": constant_features,
                "high_cardinality_features": high_cardinality,
                "potential_id_columns": potential_ids,
            }
            
        except Exception as e:
            return {"applicable": True, "error": str(e)}
    
    def identify_risks(
        self,
        dataset_info: dict,
        data_quality: dict
    ) -> list[str]:
        """
        Identify potential risks based on dataset analysis.
        
        Args:
            dataset_info: Output from analyze_dataset
            data_quality: Output from check_data_quality
            
        Returns:
            List of risk descriptions
        """
        risks = []
        
        # Sample size risks
        n_samples = dataset_info.get("n_samples", 0)
        n_features = dataset_info.get("n_features", 0)
        
        if n_samples < 100:
            risks.append("⚠️ Very small dataset (<100 samples) - high overfitting risk")
        elif n_samples < 1000:
            risks.append("⚠️ Small dataset (<1000 samples) - complex models may overfit")
        
        if n_features > n_samples:
            risks.append("⚠️ More features than samples - consider dimensionality reduction")
        
        # Missing data risks
        if data_quality.get("applicable"):
            missing_pct = data_quality.get("missing_percentage", 0)
            if missing_pct > 30:
                risks.append(f"⚠️ High missing data ({missing_pct}%) - may need imputation strategy")
            elif missing_pct > 10:
                risks.append(f"⚠️ Moderate missing data ({missing_pct}%)")
            
            if data_quality.get("constant_features"):
                risks.append("⚠️ Constant features detected - should be removed")
            
            if data_quality.get("potential_id_columns"):
                risks.append("⚠️ Potential identifier columns detected - may cause leakage")
        
        return risks
    
    def recommend_models(
        self,
        dataset_info: dict,
        modality: str
    ) -> list[str]:
        """
        Recommend models based on dataset characteristics.
        
        Args:
            dataset_info: Output from analyze_dataset
            modality: Data modality
            
        Returns:
            List of recommended model names
        """
        n_samples = dataset_info.get("n_samples", 0)
        task_type = dataset_info.get("task_type", "unknown")
        
        if modality == "tabular":
            if n_samples < 1000:
                return ["logistic_regression", "random_forest", "xgboost"]
            else:
                return ["lightgbm", "xgboost", "catboost", "random_forest"]
        
        elif modality == "timeseries":
            if n_samples < 500:
                return ["arima", "exponential_smoothing", "prophet"]
            else:
                return ["prophet", "lightgbm", "lstm", "temporal_fusion_transformer"]
        
        elif modality == "vision":
            return ["efficientnet", "resnet", "mobilenet", "vit"]
        
        elif modality == "nlp":
            if n_samples < 1000:
                return ["tfidf_logistic", "naive_bayes", "distilbert"]
            else:
                return ["bert", "roberta", "deberta"]
        
        return ["lightgbm", "random_forest"]
    
    def suggest_preprocessing(
        self,
        dataset_info: dict,
        data_quality: dict
    ) -> list[str]:
        """
        Suggest preprocessing steps.
        
        Args:
            dataset_info: Output from analyze_dataset
            data_quality: Output from check_data_quality
            
        Returns:
            List of preprocessing suggestions
        """
        steps = []
        
        if data_quality.get("applicable"):
            if data_quality.get("columns_with_missing"):
                steps.append("Handle missing values (imputation or removal)")
            
            if data_quality.get("constant_features"):
                steps.append(f"Remove constant features: {data_quality['constant_features']}")
            
            if data_quality.get("potential_id_columns"):
                steps.append(f"Consider removing ID columns: {data_quality['potential_id_columns']}")
        
        if dataset_info.get("categorical_features", 0) > 0:
            steps.append("Encode categorical features (one-hot or target encoding)")
        
        if dataset_info.get("numeric_features", 0) > 0:
            steps.append("Scale numerical features (StandardScaler or MinMaxScaler)")
        
        task_type = dataset_info.get("task_type", "")
        if "classification" in task_type:
            steps.append("Check for class imbalance (consider SMOTE or class weights)")
        
        return steps
