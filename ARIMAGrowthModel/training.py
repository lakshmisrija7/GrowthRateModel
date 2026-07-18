import pandas as pd
from .architecture import ARIMAModelArchitecture
from .logger import get_logger
from .exceptions import ValuationError

logger = get_logger(__name__)

class ARIMATrainer:
    def __init__(self, model_arch: ARIMAModelArchitecture):
        self.model_arch = model_arch

    def train_model(self, df: pd.DataFrame, feature_cols: list):
        if df.empty:
            raise ValuationError("Training dataframe is empty")
        
        logger.info("Starting ARIMA model training workflow")
        endog = df["growth"]
        self.model_arch.fit_model(endog, df, feature_cols)
        logger.info("ARIMA model training workflow completed successfully")
