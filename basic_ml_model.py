import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def get_data():
    path = 'winequality-white.csv'
    try:
        data = pd.read_csv(path, sep=";")
    except Exception as e:
        raise e
    return data

def data_split(data):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2)
    return X_train, X_test, y_train, y_test
mlflow.set_experiment(experiment_name='Usman')
def train_predict_model(X_t, y_t, X_tes, y_tes):
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=50, criterion='gini')
        model.fit(X_t, y_t)
        y_pred = model.predict(X_tes)
        score = accuracy_score(y_tes, y_pred)
        
        # Getting the predicted probabilities
        pred_prob = model.predict_proba(X_tes)
        
        
        
        roc_score = roc_auc_score(y_tes,pred_prob, multi_class='ovr')
        
        # Logging parameters and metrics
        mlflow.log_param('n_estimators', model.n_estimators)
        mlflow.log_param('max_depth', model.max_depth)
        mlflow.log_metric('score', score)
        mlflow.log_metric('roc_score', roc_score)
        
        # Logging the model itself
        mlflow.sklearn.log_model(model, 'Random Forest')

    return score

def main():
    df = get_data()
    X_train, X_test, y_train, y_test = data_split(df)
    score = train_predict_model(X_train, y_train, X_test, y_test)
    print(f"Model accuracy is {score:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
