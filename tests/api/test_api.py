import unittest
from fastapi.testclient import TestClient
from challenge import app
import numpy as np
from unittest.mock import patch


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('sklearn.linear_model._logistic.LogisticRegression.predict', return_value=np.array([0]))      
    def test_should_get_predict(self, mock_predict):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    
    @patch('sklearn.linear_model._logistic.LogisticRegression.predict', return_value=np.array([0]))  
    def test_should_failed_unkown_column_1(self, mock_predict):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    @patch('sklearn.linear_model._logistic.LogisticRegression.predict', return_value=np.array([0]))  
    def test_should_failed_unkown_column_2(self, mock_predict):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
    
    @patch('sklearn.linear_model._logistic.LogisticRegression.predict', return_value=np.array([0]))  
    def test_should_failed_unkown_column_3(self, mock_predict):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)