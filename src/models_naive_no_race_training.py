"""
No-race training: trains Logistic Regression and XGBoost after dropping race columns.
"""
import pandas as pd
from config import DATA_PATH, TRAINING_NO_RACE_OUTPUT
from data_processing import DataPreprocessor
from training.pipeline import TrainingConfig, TrainingPipeline

# Load & preprocess
raw_df = pd.read_csv(DATA_PATH)
mlready_df, _ = DataPreprocessor(raw_df=raw_df).preprocess()

# Run pipeline (drop race features)
config = TrainingConfig(output=TRAINING_NO_RACE_OUTPUT, drop_race_features=True)
pipeline = TrainingPipeline(mlready_df, config)
pipeline.run()
