"""
Naive training: trains Logistic Regression and XGBoost on the full feature set
(including race columns).
"""
import pandas as pd
from config import DATA_PATH, TRAINING_OUTPUT
from data_processing import DataPreprocessor
from training.pipeline import TrainingConfig, TrainingPipeline

# Load & preprocess
raw_df = pd.read_csv(DATA_PATH)
mlready_df, _ = DataPreprocessor(raw_df=raw_df).preprocess()

# Run pipeline
config = TrainingConfig(output=TRAINING_OUTPUT, drop_race_features=False)
pipeline = TrainingPipeline(mlready_df, config)
pipeline.run()
