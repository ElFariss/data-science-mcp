"""EDA (Exploratory Data Analysis) modules."""

from src.eda.pipeline import run_eda_pipeline
from src.eda.tabular_eda import TabularEDA
from src.eda.timeseries_eda import TimeSeriesEDA
from src.eda.vision_eda import VisionEDA
from src.eda.nlp_eda import NLPEDA

__all__ = [
    "run_eda_pipeline",
    "TabularEDA",
    "TimeSeriesEDA",
    "VisionEDA",
    "NLPEDA",
]
