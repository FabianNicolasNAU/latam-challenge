import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Tuple, Union, List
from datetime import datetime


class DelayModel:
    def __init__(self):
        """
        Initialize the DelayModel with a logistic regression pipeline.
        """
        self._model = Pipeline([
            ('classifier', LogisticRegression(max_iter=10000))
        ])

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Preprocess input data for model prediction.

        :param data: Raw data to preprocess.
        :param target_column: Name of the target column, if present in the data.
        :return: Processed features and optionally, target labels.
        """
        # Convert categorical columns to one-hot encoded columns
        features_data = pd.get_dummies(data, columns=['OPERA', 'TIPOVUELO', 'MES'])

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

        # Ensure all top_10_features are present in features_data
        for col in top_10_features:
            if col not in features_data.columns:
                features_data[col] = 0

        # Select the top 10 features
        features = features_data[top_10_features]

        if target_column:
            # Calculate time difference in minutes
            def get_min_diff(row):
                fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
                fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
                return ((fecha_o - fecha_i).total_seconds()) / 60
            
            data['min_diff'] = data.apply(get_min_diff, axis=1)
            threshold_in_minutes = 15
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            target = data['delay'].to_frame()

        return (features, target) if target_column else features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Train the model using the provided features and target.

        :param features: Input features for training.
        :param target: Target labels for training.
        """
        # Calculate class weights to balance the data
        target_series = target.iloc[:, 0]
        n_y0 = len(target_series[target_series == 0])
        n_y1 = len(target_series[target_series == 1])
        class_weight = {1: n_y0 / len(target_series), 0: n_y1 / len(target_series)}
        
        # Update class weights and train the model
        self._model.named_steps['classifier'].set_params(class_weight=class_weight)
        self._model.fit(features, target.values.ravel())

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict the delay labels for the given features.

        :param features: Input features for prediction.
        :return: Predicted delay labels.
        """
        return self._model.predict(features).tolist()
