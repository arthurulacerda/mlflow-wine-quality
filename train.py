# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def run_elastic_net(train_x, test_x, train_y, test_y, alpha, l1_ratio):
    with mlflow.start_run(run_name="Elastic_a=" + str(alpha) + "_lr=" + str(l1_ratio)):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("model", "")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")

def run_huber_regression(train_x, test_x, train_y, test_y, alpha, epsilon, max_iter):
    with mlflow.start_run(run_name="Huber_a=" + str(alpha) + "_e=" + str(epsilon) + "_mi=" + str(max_iter)):
        lr = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=max_iter)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("HuberRegressor model (alpha=%f, epsilon=%f, max_iter=%d):" % (alpha, epsilon, max_iter))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winequality-red.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alphas = [0.5, 0.75, 1]
    l1_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            run_elastic_net(train_x, test_x, train_y, test_y, alpha, l1_ratio)

    alphas = [0.0001, 0.001]
    epsilons = [1, 1.35, 1.5]
    max_iters = [125, 100]

    for alpha in alphas:
        for epsilon in epsilons:
            for max_iter in max_iters:
                run_huber_regression(train_x, test_x, train_y, test_y, alpha, epsilon, max_iter)


