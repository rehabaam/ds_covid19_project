# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.svm import SVC


def train_model(X_train, y_train, model_type="Logistic Regression"):
    """
    train_model Trains a model on the data

    Input:
    X_train: np.array: Features
    y_train: np.array: Labels
    model_type: str: Type of model to train

    Output:
    model: model: Trained model
    """

    match model_type:
        case "Logistic Regression":
            model = LogisticRegression()
        case "Linear Regression":
            model = LinearRegression()
        case "SVM Linear":
            model = SVC(kernel="linear", probability=True)
        case "SVM RBF":
            model = SVC(kernel="rbf")
        case "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        case _:
            raise ValueError("Invalid model type")
    return model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test, model_type="Logistic Regression"):
    """
    evaluate_model Evaluates a model on the data

    Input:
    model: model: Trained model
    X_test: np.array: Features
    y_test: np.array: Labels
    model_type: str: Type of model to train

    Output:
    accuracy: float: Accuracy of the model
    classification_report: str: Classification report
    """

    match model_type:
        case "Logistic Regression":
            y_pred = model.predict(X_test)
        case "Linear Regression":
            y_pred_binary = model.predict(X_test)
            y_pred = (y_pred_binary > 0.5).astype(int)  # Threshold at 0.5

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"R-squared (R²) Score: {r2:.4f}")
        case "SVM":
            y_pred = model.predict(X_test)
        case "Random Forest":
            y_pred = model.predict(X_test)
        case _:
            raise ValueError("Invalid model type")

    return accuracy_score(y_test, y_pred), classification_report(
        y_test, y_pred
    )
