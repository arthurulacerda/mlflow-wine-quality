# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    accuracy  = accuracy_score(actual.values, pred)
    recall    = recall_score(actual.values, pred)
    precision = precision_score(actual.values, pred)
    f1        = f1_score(actual.values, pred)
    return accuracy, recall, precision, f1

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winequality-red.csv")
    wine = pd.read_csv(wine_path)

    bins = (0, 6.5, 10)
    group_names = ['ruim', 'bom']
    wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
    label_quality = LabelEncoder()
    wine['quality'] = label_quality.fit_transform(wine['quality'])

    print(wine['quality'].value_counts())

    X = wine.drop('quality', axis = 1)
    y = wine['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    with mlflow.start_run():
        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(X_train, y_train)
        predicted_qualities = rfc.predict(X_test)

        (accuracy, recall, precision, f1) = eval_metrics(y_test, predicted_qualities)

        print("RandomForestClassifier:")
        print("  accuracy: %s" % accuracy)
        print("  recall: %s" % recall)
        print("  precision: %s" % precision)
        print("  f1: %s" % f1)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(rfc, "model")
