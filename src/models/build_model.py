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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
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


def train_basic_supervised_model(X_train, y_train, model_type="Logistic Regression"):
    """
    train_basic_supervised_model Trains a model on the data

    Input:
    X_train: np.array: Features
    y_train: np.array: Labels
    model_type: str: Type of model to train

    Output:
    model: model: Trained model
    """

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    print(f"Computed Class Weights:{class_weight_dict} labels: {np.unique(y_train)}")

    match model_type:
        case "Logistic Regression":
            model = LogisticRegression(
                C=0.1,
                class_weight=class_weight_dict,
                max_iter=100,
                penalty="l1",
                solver="liblinear",
            )
        case "Linear Regression":
            sample_weights = np.array([class_weight_dict[label] for label in y_train])
            model = LinearRegression()
            return model.fit(X_train, y_train, sample_weight=sample_weights)
        case "SVM Linear":
            model = SVC(kernel="linear", class_weight=class_weight_dict, probability=True)
        case "SVM RBF":
            model = SVC(kernel="rbf", class_weight=class_weight_dict)
        case "Random Forest":
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight=class_weight_dict,
            )
        case "CatBoost":
            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                class_weights=class_weight_dict,
                loss_function="Logloss",
                verbose=100,
            )
        case "CatBoost_Multi":
            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                class_weights=class_weight_dict,
                loss_function="MultiClass",
                verbose=100,
            )
        case _:
            raise ValueError("Invalid model type")
    return model.fit(X_train, y_train)


def train_advanced_supervised_model(
    X_train,
    y_train,
    image_size,
    epochs,
    num_classes,
    class_weight,
    model_type="CNN",
    classification_type="binary",
):
    """
    train_advanced_supervised_model Trains a model on the data

    Input:
    X_train: np.array: Features
    y_train: np.array: Labels
    image_size: int: Size of the image
    epochs: int: Number of epochs to train
    num_classes: int: Number of classes
    class_weight: dict: Class weights for the model
    model_type: str: Type of model to train
    classification_type: str: Type of the classification

    Output:
    model: model: Trained model
    history: history: Training history
    """

    # Set activation and loss based on class mode
    if classification_type == "binary":
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activation = "softmax"
        loss = "categorical_crossentropy"

    match model_type:
        case "CNN":
            inputs = Input(shape=(image_size, image_size, 1))
            x = Resizing(256, 256)(inputs)
            x = Rescaling(1.0 / 255)(x)

            # 🔹 First Convolution Block
            x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
            x = BatchNormalization()(x)  # Normalization improves training stability
            x = MaxPooling2D((2, 2))(x)

            # 🔹 Second Convolution Block
            x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)

            # 🔹 Third Convolution Block
            x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)

            # 🔹 Fourth Convolution Block (Extra Layers for Deeper Model)
            x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)

            # 🔹 Fifth Convolution Block (Extra Layers for Deeper Model)
            x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)

            # 🔹 Flatten & Fully Connected Layers
            x = Flatten()(x)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(256, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(128, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(64, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(32, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(16, activation="relu")(x)
            x = Dropout(0.2)(x)

            outputs = Dense(num_classes, activation=activation)(x)

            model = Model(inputs=inputs, outputs=outputs)

            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss=loss,
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

            output = Dense(num_classes, activation=activation)(x)  # Binary classification

            # Define the final model
            model = Model(inputs=base_model.input, outputs=output)

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss=loss,
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case _:
            raise ValueError("Invalid model type")

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    return model, model.fit(
        X_train,
        validation_data=y_train,
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[reduce_lr, early_stop],
    )


def evaluate_model(
    description,
    model,
    X_test,
    y_test,
    model_type="Logistic Regression",
    classification_type="binary",
):
    """
    evaluate_model Evaluates a model on the data

    Input:
    description: str: Description of the dataset
    model: model: Trained model
    X_test: np.array: Features
    y_test: np.array: Labels
    model_type: str: Type of model to train
    classification_type: str: Type of the classification

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
            loss, accuracy = model.evaluate(X_test)
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
            }

            # Fetting validation data
            images, _ = next(X_test)
            log_model(
                "Advanced Supervised Models",
                "tensorflow",
                description,
                model,
                model_type,
                images,
                metrics,
                classification_type,
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
        classification_type,
    )

    return accuracy, classification_report(y_test, y_pred)
