import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV file
df = pd.read_csv('loan_data_set.csv')

# Show first few rows
print("First 5 rows of data:")
print(df.head())

# Select only numeric columns
numeric_cols = df.select_dtypes(include='number').columns

# Calculate average of a selected column (first numeric column)
selected_column = numeric_cols[0]
average_value = df[selected_column].mean()
print(f"\nAverage of '{selected_column}': {average_value}")

# 2a. Bar chart of mean values of numeric columns
means = df[numeric_cols].mean()

plt.figure()
means.plot(kind='bar')
plt.title("Mean Values of Numeric Columns")
plt.xlabel("Columns")
plt.ylabel("Mean")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2b. Scatter plot between first two numeric columns (if available)
if len(numeric_cols) >= 2:
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]

    plt.figure()
    plt.scatter(df[x_col], df[y_col])
    plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

# 2c. Heatmap of correlation matrix
corr = df[numeric_cols].corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 3. Basic insights from correlation
print("\nCorrelation Matrix:")
print(corr)

# Find strongest correlation (excluding self-correlation)
corr_unstacked = corr.abs().unstack()
corr_unstacked = corr_unstacked[corr_unstacked < 1]  # remove 1.0 diagonal
strongest_pair = corr_unstacked.idxmax()
strongest_value = corr_unstacked.max()

print(f"\nStrongest relationship is between {strongest_pair[0]} "
      f"and {strongest_pair[1]} with correlation {strongest_value:.2f}")

if strongest_value > 0.5:
    print("Insight: These two features have a strong positive relationship.")
elif strongest_value > 0.2:
    print("Insight: These two features have a moderate relationship.")
else:
    print("Insight: Most features are weakly related to each other.")
