import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


data = pd.read_csv("FPTP_statistics.csv")
data = data.drop("name", axis=1)

# Summary Statistics
summary_stats = data.describe()
print("Description of data: \n", summary_stats, end="\n\n")

# Correlation Matrix
correlation_matrix = data.corr()
print("Correlation matrix: \n", correlation_matrix, end="\n\n")

# Visualize Feature Distributions
plt.figure(figsize=(12, 10))
data.hist(figsize=(15, 12), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# Scatter plot examples
plt.figure(figsize=(8, 6))
sns.scatterplot(x="development", y="score", data=data)
plt.title("Development vs Score", fontsize=14)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x="income", y="score", data=data)
plt.title("Income vs Score", fontsize=14)
plt.show()