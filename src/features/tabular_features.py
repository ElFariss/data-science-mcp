"""Tabular Feature Engineering."""

from typing import Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


class TabularFeatureEngineer:
    """Feature engineering for tabular data."""
    
    def __init__(self, strategy: str = "basic"):
        """
        Initialize feature engineer.
        
        Args:
            strategy: Feature engineering strategy
                - basic: scaling + one-hot encoding
                - advanced: + interactions + polynomial
                - target_encoding: target-based encoding
        """
        self.strategy = strategy
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.numeric_cols = []
        self.categorical_cols = []
        self._fitted = False
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> "TabularFeatureEngineer":
        """
        Fit the feature engineer.
        
        Args:
            df: Input dataframe (features only)
            target: Optional target for target encoding
            
        Returns:
            self
        """
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric imputation
        if self.numeric_cols:
            self.imputers['numeric'] = SimpleImputer(strategy='median')
            self.imputers['numeric'].fit(df[self.numeric_cols])
            
            # Scaling
            self.scalers['standard'] = StandardScaler()
            imputed = self.imputers['numeric'].transform(df[self.numeric_cols])
            self.scalers['standard'].fit(imputed)
        
        # Categorical imputation and encoding
        if self.categorical_cols:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            self.imputers['categorical'].fit(df[self.categorical_cols])
            
            if self.strategy == "target_encoding" and target is not None:
                self._fit_target_encoding(df, target)
            else:
                self._fit_label_encoding(df)
        
        self._fitted = True
        return self
    
    def _fit_label_encoding(self, df: pd.DataFrame) -> None:
        """Fit label encoders for categorical columns."""
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Handle unknown values
            values = df[col].fillna('__MISSING__').astype(str)
            le.fit(values)
            self.encoders[col] = le
    
    def _fit_target_encoding(self, df: pd.DataFrame, target: pd.Series) -> None:
        """Fit target encoders for categorical columns."""
        for col in self.categorical_cols:
            # Compute mean target per category
            temp_df = pd.DataFrame({col: df[col], 'target': target})
            means = temp_df.groupby(col)['target'].mean().to_dict()
            global_mean = target.mean()
            self.encoders[col] = {'means': means, 'global_mean': global_mean}
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        if not self._fitted:
            raise ValueError("Must fit before transform")
        
        result = pd.DataFrame()
        
        # Process numeric features
        if self.numeric_cols:
            numeric_data = df[self.numeric_cols].copy()
            
            # Impute
            imputed = self.imputers['numeric'].transform(numeric_data)
            
            # Scale
            scaled = self.scalers['standard'].transform(imputed)
            
            for i, col in enumerate(self.numeric_cols):
                result[col] = scaled[:, i]
        
        # Process categorical features
        if self.categorical_cols:
            cat_data = df[self.categorical_cols].copy()
            
            # Impute
            imputed = self.imputers['categorical'].transform(cat_data)
            imputed_df = pd.DataFrame(imputed, columns=self.categorical_cols, index=df.index)
            
            for col in self.categorical_cols:
                if self.strategy == "target_encoding":
                    enc = self.encoders[col]
                    result[col] = imputed_df[col].map(enc['means']).fillna(enc['global_mean'])
                else:
                    le = self.encoders[col]
                    values = imputed_df[col].astype(str)
                    # Handle unseen values
                    result[col] = values.apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ 
                        else -1
                    )
        
        # Advanced: interactions
        if self.strategy == "advanced" and len(self.numeric_cols) >= 2:
            for i, col1 in enumerate(self.numeric_cols[:3]):
                for col2 in self.numeric_cols[i+1:4]:
                    result[f"{col1}_{col2}_mul"] = result[col1] * result[col2]
        
        self.feature_names = result.columns.tolist()
        return result
    
    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, target).transform(df)
    
    def get_feature_names(self) -> list[str]:
        """Get transformed feature names."""
        return self.feature_names
    
    def save_config(self, path: str) -> None:
        """Save feature engineering configuration."""
        import json
        
        config = {
            "strategy": self.strategy,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "feature_names": self.feature_names,
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> "TabularFeatureEngineer":
        """Load feature engineering configuration."""
        import json
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        fe = cls(strategy=config["strategy"])
        fe.numeric_cols = config["numeric_cols"]
        fe.categorical_cols = config["categorical_cols"]
        fe.feature_names = config["feature_names"]
        
        return fe
