import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Union, List
from datetime import datetime

class DelayModel:
    def __init__(self):
        self._model = Pipeline([
            ('preprocessor', None), 
            ('classifier', LogisticRegression(max_iter=10000))
        ])
        self.is_fitted = False

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        # Convert categorical columns to one-hot encoded columns
        features_data = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        # List of top 10 important features
        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

        # Select the top 10 features
        features = features_data[top_10_features]

        def get_min_diff(row):
            fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff

        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        target = data['delay'].to_frame()

        return (features, target) if target_column else features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Balancing the data
        target_series = target.iloc[:, 0]
        n_y0 = len(target_series[target_series == 0])
        n_y1 = len(target_series[target_series == 1])
        class_weight = {1: n_y0/len(target_series), 0: n_y1/len(target_series)}
        self._model.named_steps['classifier'].set_params(class_weight=class_weight)
        
        # Train the model
        self._model.fit(features, target)
        self.is_fitted = True

    def predict(self, features: pd.DataFrame) -> List[int]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Please fit the model before predicting.")
        # Get predictions from the model
        predictions = self._model.predict(features)
        predictions = predictions.tolist()
        return list(predictions)
