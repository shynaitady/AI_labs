import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


# =========================
# 1. Load dataset
# =========================

data = pd.read_csv("AB_NYC_2019.csv")

print("Dataset information:")
print(data.info())

print("\nDataset description:")
print(data.describe())


# =========================
# 2. Data preprocessing
# =========================

# Drop columns that are not useful for prediction or contain too much text
data = data.drop(columns=[
    "id",
    "name",
    "host_id",
    "host_name",
    "last_review"
])

# Fill missing values
data = data.fillna(0)

# Convert categorical columns into numerical format
data = pd.get_dummies(data, columns=[
    "neighbourhood_group",
    "neighbourhood",
    "room_type"
], drop_first=True)

# Make sure all data is numeric
data = data.apply(pd.to_numeric, errors="coerce")
data = data.fillna(0)


# =========================
# 3. Correlation matrix
# =========================

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# =========================
# 4. Features and target
# =========================

X = data.drop(columns=["price"])
y = data["price"]


# =========================
# 5. Feature scaling
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================
# 6. Train-test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 7. Linear Regression
# =========================

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# =========================
# 8. Polynomial Regression
# =========================

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)


# =========================
# 9. Decision Tree Regressor
# =========================

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


# =========================
# 10. Random Forest Regressor + GridSearchCV
# =========================

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)


# =========================
# 11. RNN / LSTM Model
# =========================

X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

rnn_model = Sequential()
rnn_model.add(LSTM(50, activation="relu", input_shape=(X_train_rnn.shape[1], 1)))
rnn_model.add(Dense(1))

rnn_model.compile(
    optimizer=Adam(),
    loss="mean_squared_error"
)

history = rnn_model.fit(
    X_train_rnn,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

y_pred_rnn = rnn_model.predict(X_test_rnn).flatten()


# =========================
# 12. Model evaluation
# =========================

def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


models = {
    "Linear Regression": y_pred_lr,
    "Polynomial Regression": y_pred_poly,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "RNN": y_pred_rnn
}

results = {}

print("\nModel Evaluation Results:")
for name, preds in models.items():
    metrics = evaluate_model(y_test, preds)
    results[name] = metrics

    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")


# =========================
# 13. Results comparison table
# =========================

results_df = pd.DataFrame(results).T
print("\nComparison Table:")
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df["R2"])
plt.xticks(rotation=45)
plt.ylabel("R2 Score")
plt.title("Model Comparison by R2 Score")
plt.grid(True)
plt.show()


# =========================
# 14. Actual vs Predicted plots
# =========================

for name, preds in models.items():
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, preds, alpha=0.5)

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name} - Actual vs Predicted")
    plt.grid(True)
    plt.show()


# =========================
# 15. Residual distribution plots
# =========================

for name, preds in models.items():
    residuals = y_test - preds

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.title(f"{name} - Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


# =========================
# 16. Learning curves for classical models
# =========================

def plot_learning_curve(model, X_train, y_train, X_test, y_test, name):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_errors = []
    test_errors = []

    for size in train_sizes:
        X_train_shuffled, y_train_shuffled = shuffle(
            X_train,
            y_train,
            random_state=42
        )

        split_idx = int(size * len(X_train_shuffled))

        X_subset = X_train_shuffled[:split_idx]
        y_subset = y_train_shuffled[:split_idx]

        model.fit(X_subset, y_subset)

        y_train_pred = model.predict(X_subset)
        y_test_pred = model.predict(X_test)

        train_errors.append(mean_squared_error(y_subset, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_errors, label="Training MSE")
    plt.plot(train_sizes, test_errors, label="Testing MSE")
    plt.xlabel("Training Set Size Proportion")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_learning_curve(
    LinearRegression(),
    X_train,
    y_train,
    X_test,
    y_test,
    "Linear Regression"
)

plot_learning_curve(
    LinearRegression(),
    X_train_poly,
    y_train,
    X_test_poly,
    y_test,
    "Polynomial Regression"
)

plot_learning_curve(
    DecisionTreeRegressor(random_state=42),
    X_train,
    y_train,
    X_test,
    y_test,
    "Decision Tree"
)

plot_learning_curve(
    RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
    X_train,
    y_train,
    X_test,
    y_test,
    "Random Forest"
)


# =========================
# 17. RNN Loss over Epochs
# =========================

plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve - RNN")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 18. Select best model
# =========================

best_model_name = results_df["R2"].idxmax()

print("\nBest performing model:")
print(best_model_name)

print("\nBest model metrics:")
print(results_df.loc[best_model_name])