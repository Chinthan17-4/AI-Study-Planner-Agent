import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# 1. Load dataset
df = pd.read_csv("data/study_data_5000_10subjects.csv")


# 2. Select features
features = [
    "difficulty",
    "previous_score",
    "days_left",
    "completion_percent"
]

X = df[features]

# 3. Select target
y = df["study_hours"]


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Create ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 6. Train model
model.fit(X_train, y_train)


# 7. Make predictions
predictions = model.predict(X_test)


# 8. Evaluate model
mae = mean_absolute_error(y_test, predictions)

print("\n--- MODEL RESULTS ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Mean Absolute Error:", round(mae, 2))


# 9. Show predictions
results = X_test.copy()
results["actual_hours"] = y_test.values
results["predicted_hours"] = predictions

print("\n--- PREDICTIONS ---")
print(results)

joblib.dump(model, "models/study_hours_model.pkl")

print("\nModel saved to models/study_hours_model.pkl")