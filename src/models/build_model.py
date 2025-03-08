# -*- coding: utf-8 -*-
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.svm import SVC
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Rescaling,
    Resizing,
)
from tensorflow.keras.optimizers import Adam

from src.common.mlflow_manager import log_model


def train_basic_supervised_model(
    X_train, y_train, model_type="Logistic Regression"
):
    """
    train_basic_supervised_model Trains a model on the data

    Input:
    X_train: np.array: Features
    y_train: np.array: Labels
    model_type: str: Type of model to train

    Output:
    model: model: Trained model
    """

    match model_type:
        case "Logistic Regression":
            model = LogisticRegression(
                C=0.1,
                class_weight=None,
                max_iter=100,
                penalty="l1",
                solver="liblinear",
            )
        case "Linear Regression":
            model = LinearRegression()
        case "SVM Linear":
            model = SVC(kernel="linear", probability=True)
        case "SVM RBF":
            model = SVC(kernel="rbf")
        case "Random Forest":
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight=None,
            )
        case "CatBoost":
            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                verbose=100,
            )
        case "CatBoost_Multi":
            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                loss_function="MultiClass",
                verbose=100,
            )
        case _:
            raise ValueError("Invalid model type")
    return model.fit(X_train, y_train)


def train_advanced_supervised_model(
    X_train, y_train, image_size, epochs, model_type="CNN"
):
    """
    train_advanced_supervised_model Trains a model on the data

    Input:
    X_train: np.array: Features
    y_train: np.array: Labels
    X_test: np.array: Features
    y_test: np.array: Labels
    image_size: int: Size of the image
    epochs: int: Number of epochs to train
    model_type: str: Type of model to train

    Output:
    model: model: Trained model
    history: history: Training history
    """

    match model_type:
        case "CNN":
            inputs = Input(shape=(image_size, image_size, 1))
            x = Resizing(256, 256)(inputs)
            x = Rescaling(1.0 / 255)(x)

            x = Conv2D(32, (5, 5), padding="same", activation="relu")(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.2)(x)

            x = Flatten()(x)
            x = Dense(128, activation="relu")(x)

            outputs = Dense(1, activation="sigmoid")(x)

            model = Model(inputs=inputs, outputs=outputs)

            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case "CNN_Multi":
            inputs = Input(shape=(image_size, image_size, 1))
            x = Resizing(256, 256)(inputs)
            x = Rescaling(1.0 / 255)(x)

            x = Conv2D(32, (5, 5), padding="same", activation="relu")(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.2)(x)

            x = Flatten()(x)
            x = Dense(128, activation="relu")(x)

            outputs = Dense(1, activation="softmax")(x)

            model = Model(inputs=inputs, outputs=outputs)

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case "Transfer Learning":
            base_model = EfficientNetB0(
                weights="imagenet",
                include_top=False,
                input_shape=(image_size, image_size, 3),
            )
            # Freeze pre-trained layers to retain learned features
            base_model.trainable = False

            # Extract deep features
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation="relu")(x)
            x = Dropout(0.3)(x)

            output = Dense(1, activation="sigmoid")(x)  # Binary classification

            # Define the final model
            model = Model(inputs=base_model.input, outputs=output)

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case "Transfer Learning Multi":
            base_model = EfficientNetB0(
                weights="imagenet",
                include_top=False,
                input_shape=(image_size, image_size, 3),
            )
            # Freeze pre-trained layers to retain learned features
            base_model.trainable = False

            # Extract deep features
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation="relu")(x)
            x = Dropout(0.3)(x)

            output = Dense(1, activation="softmax")(x)  # Binary classification

            # Define the final model
            model = Model(inputs=base_model.input, outputs=output)

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case _:
            raise ValueError("Invalid model type")

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )
    return model, model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[reduce_lr, early_stop],
    )


def evaluate_model(
    description, model, X_test, y_test, model_type="Logistic Regression"
):
    """
    evaluate_model Evaluates a model on the data

    Input:
    description: str: Description of the dataset
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
        case "SVM Linear" | "SVM RBF":
            y_pred = model.predict(X_test)
        case "Random Forest":
            y_pred = model.predict(X_test)
        case "CatBoost":
            y_pred = model.predict(X_test)
        case "CNN" | "Transfer Learning":
            loss, accuracy = model.evaluate(X_test, y_test)
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
            }

            log_model(
                "Advanced Supervised Models",
                "tensorflow",
                description,
                model,
                model_type,
                X_test,
                metrics,
            )
            return loss, accuracy
        case _:
            raise ValueError("Invalid model type")

    # Log the model
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, y_pred),
        "accuracy": accuracy,
    }

    log_model(
        "Basic Supervised Models",
        "sklearn",
        description,
        model,
        model_type,
        X_test,
        metrics,
    )

    return accuracy, classification_report(y_test, y_pred)
