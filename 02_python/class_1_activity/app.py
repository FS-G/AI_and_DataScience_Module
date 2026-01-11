import pandas as pd
import seaborn as sns

# Load a built-in dataset from seaborn
df = sns.load_dataset('tips')

# Show the first 5 rows
print("First 5 rows:")
print(df.head())

# Check basic info
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Filter rows where total_bill > 30
print("\nBills greater than $30:")
print(df[df['total_bill'] > 30])

# Group by day and get average total bill
print("\nAverage total bill by day:")
print(df.groupby('day')['total_bill'].mean())

# Add a new column for tip percentage
df['tip_pct'] = (df['tip'] / df['total_bill']) * 100
print("\nTip percentages (first 5):")
print(df[['total_bill', 'tip', 'tip_pct']].head())

# Sort by tip percentage
print("\nTop 5 by tip percentage:")
print(df.sort_values(by='tip_pct', ascending=False).head())
