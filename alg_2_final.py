import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- ALGORITHM 2: LOGISTIC REGRESSION ---------
# ------------ PREDICTING MILITARY STRENGTH ------------

# Load the dataset
data = pd.read_csv("FPTP_statistics.csv")

# Define the features to be used for military classification
military_features = [
    "mil_tech", "general_fire", "general_shock", "general_maneuver",
    "general_siege", "force_strength", "manpower_balance", "income", "professionalism", "army_tradition"]

# Define thresholds for classification
force_strong_threshold = data["force_strength"].quantile(0.6)
force_weak_threshold = data["force_strength"].quantile(0.25)
manpower_strong_threshold = data["manpower_balance"].quantile(0.6)
manpower_weak_threshold = data["manpower_balance"].quantile(0.25)
miltech_strong_threshold = 0.5
miltech_weak_threshold = -0.5

# Function to classify military strength
def classify_military_strength(row):
    if (
        row["force_strength"] >= force_strong_threshold and
        row["manpower_balance"] >= manpower_strong_threshold and
        row["mil_tech"] >= miltech_strong_threshold
    ):
        return "Strong"
    elif (
        row["force_strength"] <= force_weak_threshold or
        row["manpower_balance"] <= manpower_weak_threshold or
        row["mil_tech"] <= miltech_weak_threshold
    ):
        return "Weak"
    else:
        return "Moderate"

# Apply the classification logic to the entire dataset
data["military_strength"] = data.apply(classify_military_strength, axis=1)

# Map the target column to numeric values for classification
strength_mapping = {"Weak": 0, "Moderate": 1, "Strong": 2}
data["military_strength"] = data["military_strength"].map(strength_mapping)

# Print value counts before splitting
print("Value counts before splitting:\n", data["military_strength"].value_counts())

# Define features (X) and target (y)
X = data[military_features]
y = data["military_strength"]

# Split the data into training and testing sets 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Print value counts after splitting
print("\nValue counts in training set:\n", y_train.value_counts())
print("\nValue counts in testing set:\n", y_test.value_counts())

# Train a Logistic Regression model for feature selection
logistic_regression_fs = LogisticRegression(random_state=1, max_iter=1000)
logistic_regression_fs.fit(X_train, y_train)

# Analyze feature importances using coefficients
feature_importances = abs(logistic_regression_fs.coef_).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": military_features,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature importance: \n", importance_df)

# Selecting the top features (those with importance > 0.2)
selected_features = importance_df[importance_df["Importance"] > 0.2]["Feature"].tolist()
print("\nSelected features for training: \n", selected_features)

# Update training and testing sets with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train and evaluate a Logistic Regression model with selected features
logistic_regression = LogisticRegression(random_state=1, max_iter=1000)
logistic_regression.fit(X_train_selected, y_train)

y_pred_logistic = logistic_regression.predict(X_test_selected)

accuracy_train_logistic = accuracy_score(y_train, logistic_regression.predict(X_train_selected))
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
report_logistic = classification_report(y_test, y_pred_logistic, target_names=["Weak", "Moderate", "Strong"])

print("\nLogistic regression training accuracy with selected features:", accuracy_train_logistic)
print("Logistic regression accuracy with selected features:", accuracy_logistic)
print("\nClassification report with logistic regression:\n", report_logistic)

# Create the comparison DataFrame
comparison_df = pd.DataFrame({
    "Actual": y_test.map({0: "Weak", 1: "Moderate", 2: "Strong"}).reset_index(drop=True),
    "Predicted": pd.Series(y_pred_logistic).map({0: "Weak", 1: "Moderate", 2: "Strong"})})

print("Comparison:\n", comparison_df)

# Prediction confusion matrix
confusion_matrix_df = pd.crosstab(
    comparison_df["Actual"], comparison_df["Predicted"],
    rownames=["Actual"], colnames=["Predicted"], margins=True, margins_name="Total")

# Visualize the prediction confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_df.iloc[:-1, :-1], annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Logistic Regression Prediction Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=16)
plt.ylabel("Actual Labels", fontsize=16)
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=["Moderate", "Strong", "Weak"], rotation=0)
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["Moderate", "Strong", "Weak"], rotation=0)
plt.tight_layout()
plt.show()

# Visualizing Feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.title("Feature Importances from Logistic Regression", fontsize=16)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Classification report as a heatmap
report = classification_report(y_test, y_pred_logistic, target_names=["Weak", "Moderate", "Strong"], output_dict=True)
report_df = pd.DataFrame(report).iloc[:-1, :3]  # Exclude "accuracy" row and "support" column

plt.figure(figsize=(8, 6))
sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("LR Classification Report Heatmap", fontsize=16)
plt.ylabel("Metrics", fontsize=14)
plt.xlabel("Classes", fontsize=14)
plt.tight_layout()
plt.show()