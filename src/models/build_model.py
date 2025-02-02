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
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Reshape,
)
from tensorflow.keras.optimizers import Adam


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


def train_advanced_supervised_model(
    X_train, y_train, X_test, y_test, X_input, epochs, model_type="CNN"
):
    """
    train_advanced_supervised_model Trains a model on the data

    Input:
    X_train: np.array: Features
    y_train: np.array: Labels
    X_test: np.array: Features
    y_test: np.array: Labels
    X_input: np.array: Features
    epochs: int: Number of epochs to train
    model_type: str: Type of model to train

    Output:
    model: model: Trained model
    history: history: Training history
    """

    if len(X_train) != 2 or len(X_input) != 2 or len(X_test) != 2:
        raise ValueError("Invalid input shape")

    match model_type:
        case "CNN":
            # Image Model
            input_img = Input(shape=(X_input[0], X_input[0], 1))
            x = Conv2D(32, (3, 3), activation="relu", padding="same")(
                input_img
            )
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Flatten()(x)

            # Dense Features Model
            input_features = Input(shape=(X_input[1],))
            feat_dense = Dense(32, activation="relu")(input_features)
            feat_dense = Dropout(0.2)(feat_dense)

            # Merge CNN and Statistical Features
            merged = Concatenate()([x, feat_dense])
            merged = Dense(64, activation="relu")(merged)
            merged = Dropout(0.3)(merged)
            output = Dense(1, activation="sigmoid")(merged)

            # Define Model
            model = Model(inputs=[input_img, input_features], outputs=output)
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case "Transfer Learning":
            base_model = EfficientNetB0(
                weights="imagenet",
                include_top=False,
                input_shape=(X_input[0], X_input[0], 3),
            )
            # Freeze pre-trained layers to retain learned features
            base_model.trainable = False

            # Extract deep features
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            deep_features = Dense(128, activation="relu")(x)
            deep_features = Dropout(0.3)(deep_features)

            # Dense model for statistical features
            input_features = Input(shape=(X_input[1],))
            feat_dense = Dense(32, activation="relu")(input_features)
            feat_dense = Dropout(0.2)(feat_dense)

            # Merge deep features and statistical features
            merged = layers.Concatenate()([deep_features, feat_dense])
            merged = Dense(64, activation="relu")(merged)
            merged = Dropout(0.3)(merged)
            output = Dense(1, activation="sigmoid")(
                merged
            )  # Binary classification

            # Define the final model
            model = Model(
                inputs=[base_model.input, input_features], outputs=output
            )

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()

        case "Capsule Network":
            # Image Model
            input_img = Input(shape=(X_input[0], X_input[0], 1))

            # First convolutional layer
            conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(
                input_img
            )

            # Primary Capsule Layerer
            x = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
            x = Flatten()(x)
            correct_shape = x.shape[-1] // 32  # 32 is the number of capsules
            x = Reshape((32, correct_shape))(x)  # 32 capsules of 64 dimensions

            # Fully Connected Capsule Layer
            x = Dense(128, activation="relu")(x)
            x = Flatten()(x)

            img_model = Model(input_img, x)

            # Dense Features Model
            input_features = Input(shape=(X_input[1],))
            feat_dense = Dense(32, activation="relu")(input_features)
            feat_dense = Dropout(0.2)(feat_dense)

            # Merge CNN and Statistical Features
            merged = Concatenate()([img_model.output, feat_dense])
            merged = Dense(64, activation="relu")(merged)
            merged = Dropout(0.3)(merged)
            output = Dense(1, activation="sigmoid")(merged)

            # Define Model
            model = Model(
                inputs=[img_model.input, input_features], outputs=output
            )
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case _:
            raise ValueError("Invalid model type")
    return model, model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=16,
    )


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
        case "CNN" | "Transfer Learning" | "Capsule Network":
            return model.evaluate(X_test, y_test)
        case _:
            raise ValueError("Invalid model type")

    return accuracy_score(y_test, y_pred), classification_report(
        y_test, y_pred
    )
