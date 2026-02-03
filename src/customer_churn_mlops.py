import os
import random
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import dagshub

def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

def read_data():
    url = 'https://raw.githubusercontent.com/andresakmr/customer_churn_mlops/master/data/Customer-Churn-Records.csv'
    data = pd.read_csv(url)
    # Já limpando o 'Complain' para evitar leakage feature
    X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited', 'Complain'], axis=1)
    y = data["Exited"]
    return X, y

def process_data(X, y):
    # Transformando categorias (Geography, Gender, etc)
    X = pd.get_dummies(X, columns=['Geography', 'Gender', 'Card Type'], drop_first=True)
    
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    return train_test_split(X_df, y, test_size=0.3, random_state=42)

def create_model(X):
    reset_seeds()
    model = Sequential([
        InputLayer(shape=(X.shape[1],)),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid') # 1 neurônio e Sigmoid para Churn (Sim/Não)
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def config_mlflow():
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        dagshub.init(repo_owner='andresakmr', repo_name='customer_churn_mlops', mlflow=True)
    mlflow.keras.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)

def train_model(model, X_train, y_train):
    with mlflow.start_run(run_name='treino_final_unificado_local') as run:
        model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)
        
        # Registrando o modelo no Registry
        run_uri = f'runs:/{run.info.run_id}/model'
        mlflow.register_model(run_uri, 'customer_churn')

if __name__ == "__main__":
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X_train)
    config_mlflow()
    train_model(model, X_train, y_train)