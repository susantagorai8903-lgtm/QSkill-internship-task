import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('loan_data_set.csv')


print("First 5 rows of data:")
print(df.head())


numeric_cols = df.select_dtypes(include='number').columns


selected_column = numeric_cols[0]
average_value = df[selected_column].mean()
print(f"\nAverage of '{selected_column}': {average_value}")


means = df[numeric_cols].mean()

plt.figure()
means.plot(kind='bar')
plt.title("Mean Values of Numeric Columns")
plt.xlabel("Columns")
plt.ylabel("Mean")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


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


corr = df[numeric_cols].corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


print("\nCorrelation Matrix:")
print(corr)


corr_unstacked = corr.abs().unstack()
corr_unstacked = corr_unstacked[corr_unstacked < 1]  
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

