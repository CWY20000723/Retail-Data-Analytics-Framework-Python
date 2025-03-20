import pandas as pd
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="mean")

    def handle_missing_values(self, df: pd.DataFrame):
        df_filled = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
        return df_filled
