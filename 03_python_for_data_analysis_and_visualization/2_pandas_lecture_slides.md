**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*

# Complete Pandas Lecture
## From Foundations to Advanced Analysis

---

## Module 1: Foundations & Setup

### What is Pandas?
- **Data manipulation and analysis library** for Python
- Built on top of **NumPy** for numerical operations
- Essential tool in the **data science ecosystem**
- Provides **DataFrame** and **Series** data structures

### Installation & Setup
```python
# Installation
pip install pandas numpy jupyter

# Import convention
import pandas as pd
import numpy as np

# Check version
print(pd.__version__)
```

### Core Data Structures
```python
# Series - 1D labeled array
sales_series = pd.Series([176, 135, 195, 174], 
                        index=['Sale1', 'Sale2', 'Sale3', 'Sale4'])
print(sales_series)

# DataFrame - 2D labeled data structure
sales_df = pd.DataFrame({
    'Sales': [176, 135, 195, 174],
    'COGS': [292, 225, 325, 289],
    'State': ['Connecticut', 'Connecticut', 'Connecticut', 'Connecticut']
})
print(sales_df)
```

---

## Module 2: Data Structures Deep Dive

### Creating DataFrames from Different Sources
```python
# From dictionary
data_dict = {
    'Area_Code': [203, 203, 203, 203],
    'State': ['Connecticut', 'Connecticut', 'Connecticut', 'Connecticut'],
    'Sales': [176, 135, 195, 174],
    'Profit_Margin': [107, 75, 122, 105]
}
df_from_dict = pd.DataFrame(data_dict)

# From lists
columns = ['Area_Code', 'State', 'Sales', 'Profit_Margin']
data_list = [[203, 'Connecticut', 176, 107],
             [203, 'Connecticut', 135, 75]]
df_from_list = pd.DataFrame(data_list, columns=columns)

# From NumPy array
np_data = np.array([[203, 176, 107], [203, 135, 75]])
df_from_numpy = pd.DataFrame(np_data, columns=['Area_Code', 'Sales', 'Profit_Margin'])
```

### Understanding Index Objects
```python
# Default integer index
df = pd.DataFrame({'Sales': [176, 135, 195]})
print(df.index)  # RangeIndex(start=0, stop=3, step=1)

# Custom index
df_custom = pd.DataFrame({'Sales': [176, 135, 195]}, 
                        index=['Q1', 'Q2', 'Q3'])
print(df_custom.index)  # Index(['Q1', 'Q2', 'Q3'])


```

### Memory Optimization
```python
# Check memory usage
df = pd.DataFrame({
    'Area_Code': [203, 203, 203, 203],
    'State': ['Connecticut', 'Connecticut', 'Connecticut', 'Connecticut'],
    'Sales': [176, 135, 195, 174]
})
print(df.memory_usage(deep=True))

# Optimize with categories
df['State'] = df['State'].astype('category')
print(df.memory_usage(deep=True))

# Optimize numeric types
df['Area_Code'] = df['Area_Code'].astype('int16')
```

---

## Module 3: Data Import/Export

### Reading CSV Files
```python
# Basic CSV reading
sales_df = pd.read_csv('sales.csv')

# With specific options
sales_df = pd.read_csv('sales.csv', 
                      index_col=0,           # Use first column as index
                      parse_dates=['Date'],   # Parse date column
                      dtype={'Area_Code': 'int16', 'State': 'category'},
                      na_values=['', 'NULL', 'N/A'])

# Reading with custom separator
sales_df = pd.read_csv('sales.csv', sep=',', encoding='utf-8')

# Sample first few rows
print(sales_df.head())
```

### Reading Other Formats
```python
# Excel files
# sales_excel = pd.read_excel('sales.xlsx', sheet_name='Sheet1')

# JSON files
# sales_json = pd.read_json('sales.json')

# From dictionary (simulating API response)
api_data = {
    'Area_Code': [203, 203],
    'State': ['Connecticut', 'Connecticut'],
    'Sales': [176, 135],
    'Product_Type': ['Coffee', 'Coffee']
}
df_from_api = pd.DataFrame(api_data)
```

### Writing Data
```python
# Write to CSV
sales_df.to_csv('output_sales.csv', index=False)

# Write with specific options
sales_df.to_csv('sales_processed.csv', 
               index=False,
               encoding='utf-8',
               float_format='%.2f')

# Write to Excel
# sales_df.to_excel('sales_output.xlsx', sheet_name='Sales_Data', index=False)
```

---

## Module 4: Data Exploration & Inspection

### Essential Exploration Methods
```python
# Load sample data
data = {
    'Area_Code': [203, 203, 203, 203, 203],
    'State': ['Connecticut', 'Connecticut', 'Connecticut', 'Connecticut', 'Connecticut'],
    'Market': ['East', 'East', 'East', 'East', 'East'],
    'Sales': [176, 135, 195, 174, 135],
    'COGS': [292, 225, 325, 289, 223],
    'Profit_Margin': [107, 75, 122, 105, 104],
    'Product_Type': ['Coffee', 'Coffee', 'Coffee', 'Coffee', 'Coffee']
}
sales_df = pd.DataFrame(data)

# Basic exploration
print("First 3 rows:")
print(sales_df.head(3))

print("\nLast 2 rows:")
print(sales_df.tail(2))

print("\nDataFrame info:")
print(sales_df.info())

print("\nShape (rows, columns):")
print(sales_df.shape)
```

### Statistical Summaries
```python
# Describe numeric columns
print("Statistical summary:")
print(sales_df.describe())

# Describe all columns
print("\nAll columns summary:")
print(sales_df.describe(include='all'))

# Data types
print("\nData types:")
print(sales_df.dtypes)

# Memory usage
print("\nMemory usage:")
print(sales_df.memory_usage(deep=True))
```

### Value Analysis
```python
# Unique values in each column
print("Unique values per column:")
for col in sales_df.columns:
    print(f"{col}: {sales_df[col].nunique()} unique values")

# Value counts for categorical data
print("\nProduct Type distribution:")
print(sales_df['Product_Type'].value_counts())

print("\nState distribution:")
print(sales_df['State'].value_counts())

# Null values check
print("\nMissing values:")
print(sales_df.isnull().sum())
```

### Quick Visualization
```python
# Basic plotting (requires matplotlib)
import matplotlib.pyplot as plt

# Sales distribution
sales_df['Sales'].hist(bins=10, title='Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Box plot for outlier detection
sales_df.boxplot(column='Sales')
plt.title('Sales Box Plot')
plt.show()
```

---

## Module 5: Data Selection & Indexing

### Label-based Selection with .loc
```python
# Create sample DataFrame
data = {
    'Area_Code': [203, 203, 203, 203, 203],
    'State': ['Connecticut', 'Connecticut', 'Connecticut', 'Connecticut', 'Connecticut'],
    'Sales': [176, 135, 195, 174, 135],
    'COGS': [292, 225, 325, 289, 223],
    'Product_Type': ['Coffee', 'Coffee', 'Coffee', 'Tea', 'Espresso']
}
sales_df = pd.DataFrame(data)

# Select single row by index
print("Row at index 0:")
print(sales_df.loc[0])

# Select multiple rows
print("\nRows 0 to 2:")
print(sales_df.loc[0:2])

# Select specific columns
print("\nSales and COGS columns:")
print(sales_df.loc[:, ['Sales', 'COGS']])

# Select rows and columns
print("\nFirst 3 rows, Sales and COGS:")
print(sales_df.loc[0:2, ['Sales', 'COGS']])
```

### Position-based Selection with .iloc
```python
# Select by position
print("First row:")
print(sales_df.iloc[0])

# Select range of rows
print("\nFirst 3 rows:")
print(sales_df.iloc[0:3])

# Select specific columns by position
print("\nFirst 3 rows, columns 2 and 3:")
print(sales_df.iloc[0:3, 2:4])

# Select non-contiguous rows and columns
print("\nRows 0,2,4 and columns 0,2:")
print(sales_df.iloc[[0,2,4], [0,2]])
```

### Boolean Indexing
```python
# Simple boolean condition
high_sales = sales_df['Sales'] > 150
print("High sales records:")
print(sales_df[high_sales])

# Multiple conditions
coffee_high_sales = (sales_df['Product_Type'] == 'Coffee') & (sales_df['Sales'] > 150)
print("\nCoffee with high sales:")
print(sales_df[coffee_high_sales])

# Using query method
print("\nUsing query method:")
print(sales_df.query('Sales > 150 and Product_Type == "Coffee"'))

# Complex queries
print("\nComplex query:")
print(sales_df.query('Sales > COGS * 0.5'))
```

### Index Operations
```python
# Set custom index
sales_indexed = sales_df.set_index('Product_Type')
print("DataFrame with Product_Type as index:")
print(sales_indexed)

# Reset index
sales_reset = sales_indexed.reset_index()
print("\nReset index:")
print(sales_reset)

# Multi-level indexing
sales_multi = sales_df.set_index(['State', 'Product_Type'])
print("\nMulti-level index:")
print(sales_multi)

# Access data using multi-level index
print("\nAccessing Tea data for Connecticut:")
print(sales_multi.loc[('Connecticut', 'Tea')])
```

---

## Module 6: Data Cleaning & Preprocessing

### Handling Missing Data
```python
# Create DataFrame with missing values for demonstration
import numpy as np

data = {
    'Area_Code': [203, 203, 203, np.nan, 203],
    'State': ['Connecticut', 'Connecticut', None, 'Connecticut', 'Connecticut'],
    'Sales': [176, np.nan, 195, 174, 135],
    'COGS': [292, 225, 325, 289, np.nan],
    'Product_Type': ['Coffee', 'Coffee', 'Coffee', 'Tea', 'Espresso']
}
sales_with_nulls = pd.DataFrame(data)

# Detect missing values
print("Missing values per column:")
print(sales_with_nulls.isnull().sum())

print("\nRows with any missing values:")
print(sales_with_nulls[sales_with_nulls.isnull().any(axis=1)])

# Remove missing values
print("\nDrop rows with any null values:")
cleaned_df = sales_with_nulls.dropna()
print(cleaned_df)

# Fill missing values
print("\nFill missing values:")
filled_df = sales_with_nulls.fillna({
    'Area_Code': 203,
    'State': 'Unknown',
    'Sales': sales_with_nulls['Sales'].mean(),
    'COGS': sales_with_nulls['COGS'].median()
})
print(filled_df)
```

### Data Type Conversions
```python
# Original data
data = {
    'Area_Code': ['203', '203', '203'],
    'Sales': ['176.0', '135.5', '195.2'],
    'Product_Type': ['Coffee', 'Coffee', 'Tea'],
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03']
}
df_mixed = pd.DataFrame(data)

print("Original dtypes:")
print(df_mixed.dtypes)

# Convert data types
df_converted = df_mixed.copy()
df_converted['Area_Code'] = df_converted['Area_Code'].astype('int16')
df_converted['Sales'] = df_converted['Sales'].astype('float32')
df_converted['Product_Type'] = df_converted['Product_Type'].astype('category')
df_converted['Date'] = pd.to_datetime(df_converted['Date'])

print("\nConverted dtypes:")
print(df_converted.dtypes)
```

### String Operations and Text Cleaning
```python
# Sample data with text issues
data = {
    'Product_Type': ['  Coffee  ', 'COFFEE', 'coffee', 'Tea', 'ESPRESSO'],
    'State': ['Connecticut', 'connecticut', 'CONNECTICUT', 'Connecticut', 'Connecticut'],
    'Sales': [176, 135, 195, 174, 135]
}
messy_df = pd.DataFrame(data)

# String cleaning
messy_df['Product_Type_Clean'] = (messy_df['Product_Type']
                                 .str.strip()           # Remove whitespace
                                 .str.lower()           # Convert to lowercase
                                 .str.title())          # Title case

messy_df['State_Clean'] = (messy_df['State']
                          .str.lower()
                          .str.title())

print("Cleaned data:")
print(messy_df[['Product_Type_Clean', 'State_Clean', 'Sales']])

# String contains operations
coffee_products = messy_df[messy_df['Product_Type_Clean'].str.contains('Coffee')]
print("\nCoffee products:")
print(coffee_products)
```

### Duplicate Detection and Removal
```python
# Create data with duplicates
data = {
    'Area_Code': [203, 203, 203, 203, 203],
    'State': ['Connecticut', 'Connecticut', 'Connecticut', 'Connecticut', 'Connecticut'],
    'Sales': [176, 135, 195, 176, 135],  # Duplicate values
    'Product_Type': ['Coffee', 'Coffee', 'Tea', 'Coffee', 'Coffee']
}
df_with_dups = pd.DataFrame(data)

# Check for duplicates
print("Duplicate rows:")
print(df_with_dups.duplicated())

print("\nDuplicate rows (show all):")
print(df_with_dups[df_with_dups.duplicated(keep=False)])

# Remove duplicates
df_no_dups = df_with_dups.drop_duplicates()
print("\nAfter removing duplicates:")
print(df_no_dups)

# Remove duplicates based on specific columns
df_unique_sales = df_with_dups.drop_duplicates(subset=['Sales', 'Product_Type'])
print("\nUnique Sales-Product combinations:")
print(df_unique_sales)
```

---

## Module 7: Data Transformation

### Column Operations and Calculations
```python
# Sample sales data
data = {
    'Area_Code': [203, 203, 203, 203, 203],
    'Sales': [176, 135, 195, 174, 135],
    'COGS': [292, 225, 325, 289, 223],
    'Marketing': [69, 60, 73, 69, 56],
    'Product_Type': ['Coffee', 'Coffee', 'Coffee', 'Tea', 'Espresso']
}
sales_df = pd.DataFrame(data)

# Basic calculations
sales_df['Gross_Profit'] = sales_df['Sales'] - sales_df['COGS']
sales_df['Profit_Margin_Pct'] = (sales_df['Gross_Profit'] / sales_df['Sales']) * 100
sales_df['Total_Costs'] = sales_df['COGS'] + sales_df['Marketing']

print("DataFrame with calculated columns:")
print(sales_df)

# Conditional calculations
sales_df['Performance'] = np.where(sales_df['Profit_Margin_Pct'] > 0, 'Profitable', 'Loss')
print("\nWith performance indicator:")
print(sales_df[['Sales', 'Profit_Margin_Pct', 'Performance']])
```

### Apply, Map, and Lambda Functions
```python
# Apply function to calculate profit category
def profit_category(margin):
    if margin > 10:
        return 'High'
    elif margin > 0:
        return 'Medium'
    else:
        return 'Low'

# Apply to a single column (works on each value in the column)
sales_df['Profit_Category'] = sales_df['Profit_Margin_Pct'].apply(profit_category)

# Using lambda functions (anonymous functions for simple operations)
sales_df['Sales_Category'] = sales_df['Sales'].apply(lambda x: 'High' if x > 150 else 'Low')

# Apply to entire rows using axis=1
def calculate_roi(row):
    # 'row' is a pandas Series containing all columns for that row
    return (row['Sales'] - row['Total_Costs']) / row['Total_Costs'] * 100

# axis=1 means apply function across columns (to each row)
# axis=0 (default) would apply function down columns (to each column)
sales_df['ROI'] = sales_df.apply(calculate_roi, axis=1)

print("DataFrame with applied functions:")
print(sales_df[['Sales', 'Profit_Margin_Pct', 'Profit_Category', 'Sales_Category', 'ROI']])
```

### Creating New Features
```python
# Binning continuous variables
sales_df['Sales_Bins'] = pd.cut(sales_df['Sales'], 
                               bins=[0, 140, 180, 200], 
                               labels=['Low', 'Medium', 'High'])

# Create dummy variables
product_dummies = pd.get_dummies(sales_df['Product_Type'], prefix='Product')
sales_extended = pd.concat([sales_df, product_dummies], axis=1)

print("DataFrame with new features:")
print(sales_extended[['Sales', 'Sales_Bins', 'Product_Coffee', 'Product_Tea']])
```

---

## Module 8: Grouping & Aggregation

### GroupBy Operations
```python
# Extended sample data
data = {
    'State': ['Connecticut', 'Connecticut', 'Connecticut', 'New York', 'New York', 'New York'],
    'Product_Type': ['Coffee', 'Tea', 'Espresso', 'Coffee', 'Tea', 'Espresso'],
    'Sales': [176, 54, 84, 200, 65, 95],
    'COGS': [292, 90, 144, 350, 110, 160],
    'Marketing': [69, 21, 83, 75, 25, 85],
    'Profit_Margin': [107, 33, 1, 120, 40, 10]
}
sales_df = pd.DataFrame(data)

# Basic groupby
print("Sales by State:")
state_sales = sales_df.groupby('State')['Sales'].sum()
print(state_sales)

print("\nSales by Product Type:")
product_sales = sales_df.groupby('Product_Type')['Sales'].sum()
print(product_sales)

# Multiple aggregations
print("\nMultiple aggregations by State:")
state_agg = sales_df.groupby('State').agg({
    'Sales': ['sum', 'mean', 'count'],
    'COGS': ['sum', 'mean'],
    'Profit_Margin': ['sum', 'mean']
})
print(state_agg)
```

### Advanced GroupBy Operations
```python
# Multiple grouping columns
print("Sales by State and Product Type:")
multi_group = sales_df.groupby(['State', 'Product_Type'])['Sales'].sum()
print(multi_group)

# Custom aggregation functions
def profit_ratio(series):
    return series.sum() / len(series)

custom_agg = sales_df.groupby('State').agg({
    'Sales': ['sum', 'mean'],
    'Profit_Margin': [profit_ratio, 'max']  # max returns highest profit margin within each state
})
print("\nCustom aggregation:")
print(custom_agg)

# Apply custom function to groups
def analyze_group(group):
    return pd.Series({
        'total_sales': group['Sales'].sum(),
        'avg_margin': group['Profit_Margin'].mean(),
        'best_product': group.loc[group['Sales'].idxmax(), 'Product_Type']  # Find product with highest sales in this group
    })

group_analysis = sales_df.groupby('State').apply(analyze_group)
print("\nGroup analysis:")
print(group_analysis)
```

### Pivot Tables and Cross-tabulations
```python
# Create pivot table
pivot_sales = sales_df.pivot_table(
    values='Sales',
    index='State',
    columns='Product_Type',
    aggfunc='sum',
    fill_value=0
)
print("Pivot table - Sales by State and Product:")
print(pivot_sales)

# Multiple value columns
pivot_multi = sales_df.pivot_table(
    values=['Sales', 'Profit_Margin'],
    index='State',
    columns='Product_Type',
    aggfunc='sum',
    fill_value=0
)
print("\nMultiple values pivot table:")
print(pivot_multi)

# Cross-tabulation (frequency counts with margins=True for totals)
crosstab = pd.crosstab(sales_df['State'], sales_df['Product_Type'], margins=True)
print("\nCross-tabulation:")
print(crosstab)
```

### Rolling Windows and Expanding Calculations
```python
# Time series data simulation
dates = pd.date_range('2023-01-01', periods=6, freq='D')
ts_data = {
    'Date': dates,
    'Sales': [176, 135, 195, 174, 135, 200],
    'Product_Type': ['Coffee', 'Coffee', 'Coffee', 'Tea', 'Tea', 'Espresso']
}
ts_df = pd.DataFrame(ts_data)
ts_df.set_index('Date', inplace=True)

# Rolling calculations
ts_df['Sales_3Day_Avg'] = ts_df['Sales'].rolling(window=3).mean()
ts_df['Sales_3Day_Sum'] = ts_df['Sales'].rolling(window=3).sum()

# Expanding calculations (cumulative from start to current row)
ts_df['Sales_Cumsum'] = ts_df['Sales'].expanding().sum()      # Cumulative sum
ts_df['Sales_Cumavg'] = ts_df['Sales'].expanding().mean()     # Cumulative average

print("Time series with rolling and expanding calculations:")
print(ts_df)
```

---

## Module 9: Merging & Joining Data

### Different Types of Joins
```python
# Create sample DataFrames
sales_data = {
    'ProductId': [2, 13, 5, 2, 13],
    'Sales': [176, 54, 84, 135, 33],
    'State': ['Connecticut', 'Connecticut', 'Connecticut', 'Connecticut', 'Connecticut']
}
sales_df = pd.DataFrame(sales_data)

product_data = {
    'ProductId': [2, 5, 13, 15],
    'Product_Name': ['Colombian Coffee', 'Caffe Mocha', 'Green Tea', 'Chai Latte'],
    'Category': ['Coffee', 'Espresso', 'Tea', 'Tea'],
    'Price': [12.99, 4.99, 3.99, 4.49]
}
products_df = pd.DataFrame(product_data)

print("Sales DataFrame:")
print(sales_df)
print("\nProducts DataFrame:")
print(products_df)
```

### Inner Join
```python
# Inner join - only matching records
inner_join = pd.merge(sales_df, products_df, on='ProductId', how='inner')
print("Inner Join Result:")
print(inner_join)
```

### Left Join
```python
# Left join - all records from left, matching from right
left_join = pd.merge(sales_df, products_df, on='ProductId', how='left')
print("Left Join Result:")
print(left_join)
```

### Right and Outer Joins
```python
# Right join - all records from right, matching from left
right_join = pd.merge(sales_df, products_df, on='ProductId', how='right')
print("Right Join Result:")
print(right_join)

# Outer join - all records from both
outer_join = pd.merge(sales_df, products_df, on='ProductId', how='outer')
print("\nOuter Join Result:")
print(outer_join)
```

### Merging on Multiple Keys
```python
# Sample data with multiple keys
sales_detailed = {
    'ProductId': [2, 2, 13, 13],
    'State': ['Connecticut', 'New York', 'Connecticut', 'New York'],
    'Sales': [176, 180, 54, 60],
    'Quarter': ['Q1', 'Q1', 'Q1', 'Q1']
}

pricing_data = {
    'ProductId': [2, 2, 13, 13],
    'State': ['Connecticut', 'New York', 'Connecticut', 'New York'],
    'Price': [12.99, 13.49, 3.99, 4.19],
    'Discount': [0.1, 0.05, 0.15, 0.10]
}

sales_detailed_df = pd.DataFrame(sales_detailed)
pricing_df = pd.DataFrame(pricing_data)

# Merge on multiple columns
multi_key_merge = pd.merge(sales_detailed_df, pricing_df, 
                          on=['ProductId', 'State'], 
                          how='inner')
print("Multi-key merge:")
print(multi_key_merge)
```

### Concatenating DataFrames
```python
# Sample DataFrames for concatenation
q1_sales = pd.DataFrame({
    'ProductId': [2, 13, 5],
    'Sales': [176, 54, 84],
    'Quarter': ['Q1', 'Q1', 'Q1']
})

q2_sales = pd.DataFrame({
    'ProductId': [2, 13, 5],
    'Sales': [185, 58, 92],
    'Quarter': ['Q2', 'Q2', 'Q2']
})

# Vertical concatenation (stack rows)
yearly_sales = pd.concat([q1_sales, q2_sales], ignore_index=True)
print("Concatenated sales data:")
print(yearly_sales)

# Horizontal concatenation (side by side)
additional_data = pd.DataFrame({
    'COGS': [292, 90, 144, 310, 95, 155],
    'Marketing': [69, 21, 83, 75, 25, 85]
})

complete_data = pd.concat([yearly_sales, additional_data], axis=1)
print("\nHorizontal concatenation:")
print(complete_data)
```

### Handling Overlapping Columns
```python
# DataFrames with overlapping columns
df1 = pd.DataFrame({
    'ProductId': [2, 13],
    'Sales_2023': [176, 54],
    'State': ['Connecticut', 'Connecticut']
})

df2 = pd.DataFrame({
    'ProductId': [2, 13],
    'Sales_2024': [185, 58],
    'State': ['Connecticut', 'New York']  # Different state for ProductId 13
})

# Merge with suffixes for overlapping columns
overlap_merge = pd.merge(df1, df2, on='ProductId', suffixes=('_2023', '_2024'))
print("Merge with overlapping columns:")
print(overlap_merge)
```

---

## Module 10: Time Series Analysis

### DateTime Index Creation and Manipulation
```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=10, freq='D')
ts_sales = {
    'Date': dates,
    'Sales': [176, 135, 195, 174, 135, 200, 180, 165, 190, 155],
    'Product_Type': ['Coffee'] * 5 + ['Tea'] * 5,
    'COGS': [292, 225, 325, 289, 223, 350, 300, 275, 315, 260]
}
ts_df = pd.DataFrame(ts_sales)

# Set datetime index
ts_df.set_index('Date', inplace=True)
print("Time series DataFrame:")
print(ts_df.head())

# DateTime properties
print(f"\nIndex type: {type(ts_df.index)}")
print(f"Frequency: {ts_df.index.freq}")
```

### Parsing Dates and Time Zones
```python
# Different date formats
date_strings = ['2023-01-01', '01/02/2023', '2023.01.03', '04-Jan-2023']
sales_values = [176, 135, 195, 174]

# Parse different date formats
parsed_dates = []
for date_str in date_strings:
    try:
        parsed_date = pd.to_datetime(date_str, infer_datetime_format=True)
        parsed_dates.append(parsed_date)
    except:
        parsed_dates.append(pd.NaT)

date_df = pd.DataFrame({
    'Date': parsed_dates,
    'Sales': sales_values
})
print("Parsed dates:")
print(date_df)

# Working with time zones
ts_df_utc = ts_df.copy()
ts_df_utc.index = ts_df_utc.index.tz_localize('UTC')
ts_df_est = ts_df_utc.tz_convert('US/Eastern')
print(f"\nUTC timezone: {ts_df_utc.index.tz}")
print(f"EST timezone: {ts_df_est.index.tz}")
```

### Time-based Indexing and Slicing
```python
# Create longer time series
extended_dates = pd.date_range('2023-01-01', periods=100, freq='D')
extended_sales = np.random.normal(170, 30, 100)  # Random sales data
extended_ts = pd.DataFrame({
    'Sales': extended_sales,
    'Product_Type': np.random.choice(['Coffee', 'Tea', 'Espresso'], 100)
}, index=extended_dates)

# Time-based slicing
print("January 2023 data:")
jan_data = extended_ts['2023-01']
print(jan_data.head())

print(f"\nJanuary 2023 shape: {jan_data.shape}")

# Slice by date range
print("\nFirst week of January:")
first_week = extended_ts['2023-01-01':'2023-01-07']
print(first_week)

# Boolean indexing with dates
recent_data = extended_ts[extended_ts.index > '2023-02-01']
print(f"\nData after Feb 1st: {recent_data.shape[0]} records")
```

### Resampling and Frequency Conversion
```python
# Daily to weekly resampling
weekly_sales = extended_ts.resample('W')['Sales'].agg({
    'Total_Sales': 'sum',
    'Avg_Sales': 'mean',
    'Max_Sales': 'max',
    'Sales_Count': 'count'
})
print("Weekly aggregated sales:")
print(weekly_sales.head())

# Monthly resampling
monthly_sales = extended_ts.resample('M').agg({
    'Sales': ['sum', 'mean', 'std'],
    'Product_Type': lambda x: x.value_counts().index[0]  # Most common product
})
print("\nMonthly aggregated data:")
print(monthly_sales.head())

# Upsampling (fill missing values)
hourly_data = extended_ts.resample('H').ffill()  # Forward fill
print(f"\nUpsampled to hourly: {hourly_data.shape[0]} records")
```

### Time Series Visualization (Optional)
```python
import matplotlib.pyplot as plt

# Plot daily sales
plt.figure(figsize=(12, 6))
extended_ts['Sales'].plot(title='Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Plot weekly aggregation
plt.figure(figsize=(12, 6))
weekly_sales['Total_Sales'].plot(kind='bar', title='Weekly Total Sales')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Lag Features and Moving Averages
```python
# Create lag features
extended_ts['Sales_Lag1'] = extended_ts['Sales'].shift(1)
extended_ts['Sales_Lag7'] = extended_ts['Sales'].shift(7)

# Moving averages
extended_ts['MA_7day'] = extended_ts['Sales'].rolling(window=7).mean()
extended_ts['MA_30day'] = extended_ts['Sales'].rolling(window=30).mean()

# Exponential moving average
extended_ts['EMA_7day'] = extended_ts['Sales'].ewm(span=7).mean()

print("Time series with lag features and moving averages:")
print(extended_ts[['Sales', 'Sales_Lag1', 'Sales_Lag7', 'MA_7day', 'EMA_7day']].head(10))

# Calculate day-over-day change
extended_ts['Sales_Change'] = extended_ts['Sales'].pct_change()
extended_ts['Sales_Diff'] = extended_ts['Sales'].diff()

print("\nSales changes:")
print(extended_ts[['Sales', 'Sales_Change', 'Sales_Diff']].head(10))
```

---

## Practical Exercise: Complete Sales Analysis

### Let's put it all together with a comprehensive analysis:

```python
# Step 1: Load and explore the data
sales_data = {
    'Area_Code': [203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203],
    'State': ['Connecticut'] * 16,
    'Market': ['East'] * 16,
    'Market_Size': ['Small Market'] * 16,
    'Profit_Margin': [107, 75, 122, 105, 104, 104, 135, 171, 181, 15, 33, 17, 27, 49, -2, 1],
    'Sales': [176, 135, 195, 174, 135, 135, 155, 188, 195, 31, 54, 28, 36, 54, 75, 84],
    'COGS': [292, 225, 325, 289, 223, 223, 275, 334, 346, 51, 90, 47, 64, 96, 128, 144],
    'Total_Expenses': [116, 90, 130, 115, 90, 90, 103, 125, 130, 20, 36, 19, 24, 36, 53, 60],
    'Marketing': [69, 60, 73, 69, 56, 56, 64, 73, 73, 16, 21, 15, 18, 21, 77, 83],
    'ProductId': [2, 2, 2, 2, 2, 2, 2, 2, 2, 13, 13, 13, 13, 13, 5, 5],
    'Date': pd.date_range('2023-01-01', periods=16, freq='D'),
    'Product_Type': ['Coffee'] * 9 + ['Tea'] * 5 + ['Espresso'] * 2
}

# Create comprehensive DataFrame
df = pd.DataFrame(sales_data)

print("=== COMPLETE SALES ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Step 2: Data Quality Check
print("\n=== DATA QUALITY ===")
print("Missing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df[['Sales', 'COGS', 'Profit_Margin', 'Marketing']].describe())

# Step 3: Product Performance Analysis
print("\n=== PRODUCT PERFORMANCE ===")
product_performance = df.groupby('Product_Type').agg({
    'Sales': ['count', 'sum', 'mean'],
    'Profit_Margin': ['sum', 'mean'],
    'COGS': ['sum', 'mean']
}).round(2)

print("Product performance summary:")
print(product_performance)

# Step 4: Time Series Analysis
df_ts = df.set_index('Date')
df_ts['Sales_MA3'] = df_ts['Sales'].rolling(window=3).mean()
df_ts['Profit_MA3'] = df_ts['Profit_Margin'].rolling(window=3).mean()

print("\n=== TIME SERIES TRENDS ===")
print("Sales with 3-day moving average:")
print(df_ts[['Sales', 'Sales_MA3', 'Profit_Margin', 'Profit_MA3']].tail(10))

# Step 5: Profitability Analysis
df['Profit_Ratio'] = df['Profit_Margin'] / df['Sales']
df['Cost_Ratio'] = df['COGS'] / df['Sales']
df['Marketing_Efficiency'] = df['Sales'] / df['Marketing']

profitability = df.groupby('Product_Type').agg({
    'Profit_Ratio': 'mean',
    'Cost_Ratio': 'mean',
    'Marketing_Efficiency': 'mean'
}).round(3)

print("\n=== PROFITABILITY METRICS ===")
print(profitability)

# Step 6: Identify Best and Worst Performers
print("\n=== TOP PERFORMERS ===")
best_sales_day = df.loc[df['Sales'].idxmax()]
print(f"Best sales day: {best_sales_day['Date'].strftime('%Y-%m-%d')}")
print(f"Product: {best_sales_day['Product_Type']}, Sales: ${best_sales_day['Sales']}")

worst_profit_day = df.loc[df['Profit_Margin'].idxmin()]
print(f"\nWorst profit day: {worst_profit_day['Date'].strftime('%Y-%m-%d')}")
print(f"Product: {worst_profit_day['Product_Type']}, Profit: ${worst_profit_day['Profit_Margin']}")

# Step 7: Create Summary Dashboard
print("\n=== EXECUTIVE SUMMARY ===")
total_sales = df['Sales'].sum()
total_profit = df['Profit_Margin'].sum()
avg_daily_sales = df['Sales'].mean()
profit_margin_pct = (total_profit / total_sales) * 100

print(f"Total Sales: ${total_sales:,.2f}")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Average Daily Sales: ${avg_daily_sales:.2f}")
print(f"Overall Profit Margin: {profit_margin_pct:.1f}%")
print(f"Best Product: {product_performance.loc[product_performance[('Sales', 'sum')].idxmax()].name}")

# Step 8: Export Results
summary_df = df.groupby('Product_Type').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Profit_Margin': ['sum', 'mean'],
    'COGS': ['sum', 'mean'],
    'Marketing': ['sum', 'mean']
}).round(2)

print("\n=== READY FOR EXPORT ===")
print("Summary table:")
print(summary_df)

# This would save to CSV:
# summary_df.to_csv('sales_analysis_summary.csv')
# df.to_csv('processed_sales_data.csv', index=False)
```

---

## Key Takeaways & Best Practices

### Performance Tips
- **Use vectorized operations** instead of loops
- **Choose appropriate data types** (int16 vs int64, category vs object)
- **Use `query()` method** for complex filtering
- **Leverage `groupby()` efficiently** for aggregations
- **Use `pd.concat()` instead of repeated `append()`**

### Common Pitfalls to Avoid
- **SettingWithCopyWarning**: Use `.loc[]` for assignment
- **Chained indexing**: `df['col1'][df['col2'] > 0]` → `df.loc[df['col2'] > 0, 'col1']`
- **Not handling missing data** before operations
- **Forgetting to reset index** after groupby operations
- **Not validating data types** after import

### Next Steps
1. **Practice with real datasets** from your domain
2. **Learn visualization libraries** (matplotlib, seaborn, plotly)
3. **Explore advanced topics** (multi-indexing, categorical data, sparse arrays)
4. **Integration with other tools** (scikit-learn for ML, SQLAlchemy for databases)
5. **Performance optimization** for large datasets

---

## Resources for Continued Learning
- **Official Documentation**: pandas.pydata.org
- **GitHub Examples**: github.com/pandas-dev/pandas
- **Stack Overflow**: Most common pandas questions and solutions
- **Kaggle Datasets**: Practice with real-world data
- **Books**: "Python for Data Analysis" by Wes McKinney

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*