import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


# 1. Load dataset

data = pd.read_csv("AB_NYC_2019.csv")

# 2. Basic cleaning
data = data.drop(columns=[
    "id",
    "name",
    "host_id",
    "host_name",
    "last_review"
], errors="ignore")

data = data[data["price"] > 0]

# Remove extreme outliers in price
upper_limit = data["price"].quantile(0.99)
data = data[data["price"] <= upper_limit]

# Fill missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].fillna("Unknown")
    else:
        data[col] = data[col].fillna(data[col].median())


# 3. Data visualization

plt.figure(figsize=(8, 5))
sns.histplot(data["price"], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=data["price"])
plt.title("Price Boxplot")
plt.xlabel("Price")
plt.show()


# 4. Correlation matrix
numeric_data = data.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 5. Features and target

X = data.drop(columns=["price"])
y = data["price"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# Preprocessing:
# numerical columns -> scaling
# categorical columns -> one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Evaluation function

results = {}

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    }

    print(f"\n{name}:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.2f}")


def plot_model_results(name, y_true, y_pred):
    # Actual vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--"
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name}: Actual vs Predicted")
    plt.grid(True)
    plt.show()

    # Residuals
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=40, kde=True)
    plt.title(f"{name}: Residual Distribution")
    plt.xlabel("Residuals")
    plt.grid(True)
    plt.show()


# 8. Linear Regression

linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

evaluate_model("Linear Regression", y_test, y_pred_lr)
plot_model_results("Linear Regression", y_test, y_pred_lr)

# 9. Polynomial Regression

poly_preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False))
        ]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

polynomial_model = Pipeline(steps=[
    ("preprocessor", poly_preprocessor),
    ("model", LinearRegression())
])

polynomial_model.fit(X_train, y_train)
y_pred_poly = polynomial_model.predict(X_test)

evaluate_model("Polynomial Regression", y_test, y_pred_poly)
plot_model_results("Polynomial Regression", y_test, y_pred_poly)


# 10. Decision Tree

dt_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(
        random_state=42,
        max_depth=10,
        min_samples_split=10
    ))
])

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

evaluate_model("Decision Tree", y_test, y_pred_dt)
plot_model_results("Decision Tree", y_test, y_pred_dt)


# 11. Random Forest


rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

evaluate_model("Random Forest", y_test, y_pred_rf)
plot_model_results("Random Forest", y_test, y_pred_rf)


# 12. RNN / LSTM

# For neural network we need already processed numerical data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrix to dense matrix
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()

# LSTM expects 3D input:
# samples, timesteps, features
X_train_rnn = X_train_processed.reshape(
    X_train_processed.shape[0],
    1,
    X_train_processed.shape[1]
)

X_test_rnn = X_test_processed.reshape(
    X_test_processed.shape[0],
    1,
    X_test_processed.shape[1]
)

rnn_model = Sequential()
rnn_model.add(LSTM(50, activation="relu", input_shape=(1, X_train_processed.shape[1])))
rnn_model.add(Dense(32, activation="relu"))
rnn_model.add(Dense(1))

rnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error"
)

history = rnn_model.fit(
    X_train_rnn,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

y_pred_rnn = rnn_model.predict(X_test_rnn).flatten()

evaluate_model("RNN / LSTM", y_test, y_pred_rnn)
plot_model_results("RNN / LSTM", y_test, y_pred_rnn)

plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve: RNN / LSTM")
plt.legend()
plt.grid(True)
plt.show()

# Results


results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="R2", ascending=False)

print("\nModel comparison:")
print(results_df)

results_df.to_csv("model_comparison_results.csv")

plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df["R2"])
plt.xticks(rotation=30)
plt.title("Model Comparison by R2 Score")
plt.ylabel("R2 Score")
plt.xlabel("Model")
plt.grid(True)
plt.show()

best_model = results_df.index[0]

print(f"""
Final interpretation:
The best-performing model is {best_model}.
""")