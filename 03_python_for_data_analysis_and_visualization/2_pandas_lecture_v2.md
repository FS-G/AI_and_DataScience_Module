**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*

---

# Pandas Lecture — Practical Sales Data

## Our goal

We will use the provided `data/sales.csv` file to answer simple business questions. A row is one sales record. The file has columns such as `Sales`, `COGS`, `Product`, `Product Type`, `Market`, and `Date`. In this teaching copy, a few values are missing, one row is repeated, and Profit is not included—we will calculate it from Sales and COGS.

Run the examples from the main course folder so the file path works.

## 1. Import Pandas and read the file

```python
import pandas as pd

sales = pd.read_csv("data/sales.csv")
```

`read_csv` reads a CSV file and creates a DataFrame. A DataFrame is a table with rows and columns.

## 2. Other ways to make a DataFrame

Most often, we read data from a file. We can also make a small DataFrame directly from a dictionary, lists, or a NumPy array:

```python
import numpy as np

# From a dictionary: each list becomes a column
from_dictionary = pd.DataFrame({"Product": ["Tea", "Coffee"], "Sales": [100, 150]})

# From rows stored in lists
from_lists = pd.DataFrame([["Tea", 100], ["Coffee", 150]], columns=["Product", "Sales"])

# From a NumPy array
numbers = np.array([[100], [150]])
from_array = pd.DataFrame(numbers, columns=["Sales"])
```

These are brief examples. Our lesson will continue with the real sales file.

## 3. Take a first look

```python
print(sales.head())
```

`head()` shows the first five rows. This is a quick way to check that the file loaded.

```python
print(sales.shape)
print(sales.columns)
```

`shape` tells us the number of rows and columns. `columns` shows the column names.

## 4. Check the data

```python
print(sales.info())
print(sales.isna().sum())
```

`info()` shows column names and data types. `isna().sum()` counts missing values in each column.

## 5. Handle missing values and duplicate rows

The teaching CSV has missing values in Sales and COGS, plus one repeated row. First, count the missing values:

```python
print(sales.isna().sum())
```

For practice, we can see what mean, median, and forward fill would do to Sales:

```python
sales_mean_example = sales["Sales"].fillna(sales["Sales"].mean())
sales_median_example = sales["Sales"].fillna(sales["Sales"].median())
sales_forward_example = sales["Sales"].ffill()
```

Mean uses the average; median uses the middle value; forward fill (`ffill`) copies the previous row's value. For our main analysis, we will fill missing Sales and COGS with their medians:

```python
sales["Sales"] = sales["Sales"].fillna(sales["Sales"].median())
sales["COGS"] = sales["COGS"].fillna(sales["COGS"].median())
```

We can also remove rows with missing values instead of filling them:

```python
sales_without_missing = sales.dropna()
```

Now check for and remove the repeated row:

```python
print("Repeated rows:", sales.duplicated().sum())
sales = sales.drop_duplicates()
```

`duplicated()` counts repeated rows, and `drop_duplicates()` keeps one copy of each row.

## 6. Choose columns and calculate profit

```python
print(sales["Product"])
```

One pair of square brackets and a column name selects one column.

```python
small_table = sales[["Product", "Sales", "COGS"]]
print(small_table.head())
```

Use a list of column names to select several columns.

The CSV does not contain a Profit column. We will calculate it using the business rule **Profit = Sales - COGS**:

```python
sales["Profit"] = sales["Sales"] - sales["COGS"]
print(sales[["Sales", "COGS", "Profit"]].head())
```

Pandas subtracts the values row by row and stores each result in the new Profit column.

## 7. Choose rows that match a condition

```python
high_sales = sales[sales["Sales"] > 200]
print(high_sales[["Product", "Sales", "COGS", "Profit"]].head())
```

This keeps rows where Sales is greater than 200. Try changing `200` to another amount.

## 8. Sort the records

```python
highest_sales = sales.sort_values("Sales", ascending=False)
print(highest_sales[["Product", "Sales"]].head())
```

Sorting helps us see the largest sales first. `ascending=False` means descending order.

## 9. Calculate simple summaries

```python
print("Total sales:", sales["Sales"].sum())
print("Average sale:", sales["Sales"].mean())
print("Highest sale:", sales["Sales"].max())
```

These calculations use the `Sales` column.

## 10. Compare products

```python
sales_by_product = sales.groupby("Product")["Sales"].sum()
print(sales_by_product)
```

`groupby("Product")` makes a group for each product. `.sum()` adds the sales in each group.

We can do the same for product type:

```python
sales_by_type = sales.groupby("Product Type")["Sales"].sum()
print(sales_by_type)
```

## 11. Compare sales and profit by market

```python
market_summary = sales.groupby("Market")[["Sales", "Profit"]].sum()
print(market_summary)
```

This gives one total for Sales and one for Profit in each market.

## 12. Combine DataFrames

There are two common ways to put DataFrames together:

- **Vertically:** stack rows with `pd.concat`.
- **Horizontally:** add columns side by side with `pd.concat`, or match rows using a shared key with `pd.merge`.

Here are two tiny tables for practice:

```python
jan = pd.DataFrame({"Product": ["Tea", "Coffee"], "Sales": [100, 150]})
feb = pd.DataFrame({"Product": ["Tea", "Coffee"], "Sales": [120, 170]})

# Vertical: add February rows under January rows
two_months = pd.concat([jan, feb], ignore_index=True)
print(two_months)
```

The column names should match when stacking similar rows.

### Horizontal combination by matching a key

```python
product_info = pd.DataFrame({"Product": ["Tea", "Coffee"], "Category": ["Hot drink", "Hot drink"]})
prices = pd.DataFrame({"Product": ["Tea", "Coffee"], "Price": [5, 8]})

product_table = pd.merge(product_info, prices, on="Product")
print(product_table)
```

`on="Product"` tells Pandas to match rows with the same Product value.

### Join types

Join type controls which key values are kept when the two tables do not contain exactly the same keys. Use these tables to see the difference:

```python
left_table = pd.DataFrame({"Product": ["Tea", "Coffee"], "Sales": [100, 150]})
right_table = pd.DataFrame({"Product": ["Tea", "Juice"], "Price": [5, 3]})

inner_join = pd.merge(left_table, right_table, on="Product", how="inner")
left_join = pd.merge(left_table, right_table, on="Product", how="left")
right_join = pd.merge(left_table, right_table, on="Product", how="right")
outer_join = pd.merge(left_table, right_table, on="Product", how="outer")
```

- **Inner:** keep only keys found in both tables (here, Tea).
- **Left:** keep every key from the left table; unmatched right values are missing.
- **Right:** keep every key from the right table; unmatched left values are missing.
- **Outer:** keep every key from either table; unmatched values are missing.

To inspect each result, print it, for example: `print(inner_join)`.

## 13. A simple class exercise

Use the same steps to answer:

1. What is the average profit?
2. Which product has the highest total sales?
3. What rows have sales below 100?

Hint: change the column name, condition, or aggregation in the examples above. The next section is a quick reference for other common file formats.

## Reading and writing common file types

Pandas has functions for several common file formats. CSV is our focus in this lesson. The other formats are listed here so students can recognize the options.

```python
# Read data
sales = pd.read_csv("data/sales.csv")
# excel_data = pd.read_excel("sales.xlsx")
# json_data = pd.read_json("sales.json")

# Write data
sales.to_csv("sales_copy.csv", index=False)
# sales.to_excel("sales_copy.xlsx", index=False)
# sales.to_json("sales_copy.json")
```

The Excel and JSON examples are comments, so they do not run unless you remove `#`. Reading or writing Excel files may require an additional package to be installed.

## Remember

- `pd.read_csv(...)` loads the CSV.
- `head()` gives a quick preview.
- `sales["Sales"]` selects a column.
- A condition inside `sales[...]` filters rows.
- `groupby(...)` compares categories.
