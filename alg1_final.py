import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ---------- ALGORITHM 1: DECISION TREE CLASSIFIER ---------
# --------------- PREDICTING MILITARY STRENGTH ---------------

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

# Visualizing the distribution of military_strength
plt.figure(figsize=(8, 6))
data["military_strength"].value_counts().plot(kind="bar", color=["red", "orange", "green"])
plt.title("Distribution of Military Strength", fontsize=16)
plt.xlabel("Military Strength Category", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(ticks=[0, 1, 2], labels=["Weak", "Moderate", "Strong"], rotation=0)
plt.show()

# Map the target column to numeric values for classification
strength_mapping = {"Weak": 0, "Moderate": 1, "Strong": 2}
data["military_strength"] = data["military_strength"].map(strength_mapping)

# Correlation matrix and visualization
correlation_matrix = data[military_features + ["military_strength"].copy()].corr()
print("\nCorrelation matrix:", correlation_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap="coolwarm", cbar=True, square=True,
            xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(ticks=[i + 0.5 for i in range(len(correlation_matrix.columns))], labels=correlation_matrix.columns, rotation=45, ha="center")
plt.yticks(ticks=[i + 0.5 for i in range(len(correlation_matrix.columns))], labels=correlation_matrix.columns, rotation=0, va="center")
plt.tight_layout()
plt.show()

# Define features (X) and target (y)
X = data[military_features]
y = data["military_strength"]

# Split the dataset into training and testing sets 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

print("\nValue counts in testing set:\n", y_test.value_counts())

# Train and evaluate the Decision Tree Classifier without feature selection
decision_tree = DecisionTreeClassifier(random_state=1)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

accuracy_train = accuracy_score(y_train, decision_tree.predict(X_train))
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Weak", "Moderate", "Strong"])

print("\nDecision tree classifier training accuracy without feature selection:", accuracy_train)
print("Decision tree classifier accuracy without feature selection:", accuracy)
print("\nClassification report without feature selection:\n", report)

# Analyze feature importances
feature_importances = decision_tree.feature_importances_
importance_df = pd.DataFrame({
    "Feature": military_features,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Visualizing Feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.title("Feature Importances from Decision Tree", fontsize=16)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

print("\nFeature importances: \n", importance_df)

# Selecting the top features (those with importance > 0.1)
selected_features = importance_df[importance_df["Importance"] > 0.1]["Feature"].tolist()
print("\nSelected features for training: \n", selected_features)

# Re-train the model with best features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

decision_tree_selected = DecisionTreeClassifier(random_state=1)
decision_tree_selected.fit(X_train_selected, y_train)
y_pred_selected = decision_tree_selected.predict(X_test_selected)

# Visualizing the decision tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree_selected, feature_names=selected_features, class_names=["Weak", "Moderate", "Strong"], filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization", fontsize=16)
plt.show()

accuracy_train_selected = accuracy_score(y_train, decision_tree_selected.predict(X_train_selected))
accuracy_selected = accuracy_score(y_test, y_pred_selected)
report_selected = classification_report(y_test, y_pred_selected, target_names=["Weak", "Moderate", "Strong"])

print("\nDecision tree classifier training accuracy with feature selection:", accuracy_train_selected)
print("Decision tree classifier accuracy with feature selection:", accuracy_selected)
print("\nDecision Tree Classification report with feature selection:\n", report_selected)

# Create the comparison DataFrame
comparison_df = pd.DataFrame({
    "Actual": y_test.map({0: "Weak", 1: "Moderate", 2: "Strong"}).reset_index(drop=True),
    "Predicted": pd.Series(y_pred_selected).map({0: "Weak", 1: "Moderate", 2: "Strong"})})

# Prediction confusion matrix
confusion_matrix_df = pd.crosstab(
    comparison_df["Actual"], comparison_df["Predicted"],
    rownames=["Actual"], colnames=["Predicted"], margins=True, margins_name="Total")

# Visualize the prediction confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_df.iloc[:-1, :-1], annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Decision Tree Prediction Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=16)
plt.ylabel("Actual Labels", fontsize=16)
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=["Moderate", "Strong", "Weak"], rotation=0)
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["Moderate", "Strong", "Weak"], rotation=0)
plt.tight_layout()
plt.show()

print("Model predictions: \n", comparison_df)

# Classification report as a heatmap
report_selected_dict = classification_report(y_test, y_pred_selected, target_names=["Weak", "Moderate", "Strong"], output_dict=True)
report_selected_df = pd.DataFrame(report_selected_dict).iloc[:-1, :3]  # Exclude "accuracy" row and "support" column

plt.figure(figsize=(8, 6))
sns.heatmap(report_selected_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("DT Classification Report Heatmap", fontsize=16)
plt.ylabel("Metrics", fontsize=14)
plt.xlabel("Classes", fontsize=14)
plt.tight_layout()
plt.show()
