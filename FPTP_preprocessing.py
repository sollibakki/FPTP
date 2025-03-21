import pandas as pd
from sklearn.preprocessing import StandardScaler

# Preprocessing for EU4 Data

# Reading full stats file
health_full = pd.read_csv("FPTP_health.csv")

# Reading full scores file
scores_full = pd.read_csv("FPTP_score.csv")

# Modifying data:
# Dropping the 'tag' column and sorting by 'name' in the health dataset
health_full = health_full.drop("tag", axis=1)
health_full = health_full.sort_values(by=["name"])
health_full = health_full.reset_index(drop=True)

# Dropping the 'tag' column and filtering scores for the year 1500
scores_full = scores_full.drop("tag", axis=1)
scores_full_filtered = scores_full.loc[scores_full["year"] == 1500]
scores_full_filtered = scores_full_filtered.reset_index(drop=True)

# Extracting the 'value' column as the score
scores_value = scores_full_filtered["value"]
scores_value = scores_value.rename("score")

# Printing the filtered scores for review
print(scores_full_filtered, end="\n\n")
print(scores_value, end="\n\n")

# Combining health data and scores into a single DataFrame
combined_data = pd.concat([health_full, scores_value], axis=1)

# Excluding non-numeric columns from standardization (e.g., 'name')
numeric_columns = combined_data.select_dtypes(include=["float64", "int64"]).columns
numeric_columns = numeric_columns.drop("score")  # Exclude column 'score'

# Standardizing the numeric feature columns
scaler = StandardScaler()
combined_data[numeric_columns] = scaler.fit_transform(combined_data[numeric_columns])

# Detecting outliers using Z-score method
outlier_threshold = 3  # Threshold for Z-score
for column in numeric_columns:
    z_scores = (combined_data[column] - combined_data[column].mean()) / combined_data[column].std()
    outliers = combined_data.loc[(z_scores.abs() > outlier_threshold)]
    print(f"Outliers detected in column '{column}':")
    print(outliers[[column]], end="\n\n")

# Printing the combined data for verification
print(combined_data)

# Saving the preprocessed data to a new CSV file
combined_data.to_csv("FPTP_statistics.csv", index=False)
