import pandas as pd
import pytest
from tensorflow.keras.models import Sequential
from src.customer_churn_mlops import read_data, create_model, train_model

@pytest.fixture
def sample_data():
    """Cria um mini-dataset de teste para não precisar baixar o original"""
    data = pd.DataFrame({
        'CreditScore': [600, 700],
        'Age': [40, 50],
        'Tenure': [3, 5],
        'Balance': [60000, 0],
        'NumOfProducts': [2, 1],
        'HasCrCard': [1, 1],
        'IsActiveMember': [1, 0],
        'EstimatedSalary': [50000, 100000],
        'Satisfaction Score': [3, 5],
        'Point Earned': [400, 500],
        'Geography_Germany': [0, 1],
        'Geography_Spain': [0, 0],
        'Gender_Male': [1, 0],
        'Card Type_GOLD': [0, 1],
        'Card Type_PLATINUM': [1, 0],
        'Card Type_SILVER': [0, 0],
        'Exited': [0, 1]
    })
    return data

def test_read_data():
    """Testa se a leitura do CSV no GitHub está funcionando"""
    X, y = read_data()
    assert not X.empty
    assert not y.empty

def test_create_model():
    """Testa se o modelo Keras é criado com a estrutura correta"""
    # Criamos um X fictício com 16 colunas (que é o que o modelo espera)
    import numpy as np
    X_dummy = np.zeros((1, 16))
    model = create_model(X_dummy)

    assert len(model.layers) == 3 # Input + 2 Hidden + Output (Dense)
    assert isinstance(model, Sequential)
    assert model.output_shape == (None, 1) # Saída Sigmoid (1 neurônio)

def test_train_model(sample_data):
    """Testa se o treinamento inicia sem dar erro de código"""
    X = sample_data.drop(['Exited'], axis=1)
    y = sample_data['Exited']
    model = create_model(X)
    
    # Rodamos o treino com apenas 1 época e sem logar no MLflow para ser rápido
    model.fit(X, y, epochs=1, verbose=0)
    assert model.history.history['loss'][-1] > 0