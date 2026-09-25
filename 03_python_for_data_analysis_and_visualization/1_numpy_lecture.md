**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*

---

# NumPy Lecture

## What is NumPy?

NumPy is a Python library for working with collections of numbers. Its main structure is an **array**. Arrays are useful for calculations and are also used by tools such as Pandas.

We will use small examples and learn the parts that are useful for basic data analysis.

## 1. Import NumPy

```python
import numpy as np
```

`import` makes the library available. `np` is the short name commonly used for NumPy.

## 2. Make arrays

The most common way to make an array is from a Python list:

```python
sales = np.array([100, 150, 200, 125])
print(sales)
```

An array can also have rows and columns. This is called a **2D array**:

```python
monthly_sales = np.array([[100, 150, 200],
                          [120, 180, 160]])
print(monthly_sales)
```

NumPy also has simple helpers for making arrays:

```python
print(np.zeros(3))       # Three zeros
print(np.ones(3))        # Three ones
print(np.arange(1, 6))   # Numbers from 1 up to, but not including, 6
```

## 3. Learn about an array

```python
print("Shape:", monthly_sales.shape)
print("Number of dimensions:", monthly_sales.ndim)
print("Number of values:", monthly_sales.size)
print("Data type:", monthly_sales.dtype)
```

`shape` gives the rows and columns. `ndim` tells how many dimensions the array has. `size` is the total number of values. `dtype` tells the kind of values stored, such as integers or decimals.

## 4. Select values by position

Python starts counting positions from zero.

```python
print(sales[0])    # First value
print(sales[1])    # Second value
print(sales[-1])   # Last value
```

For a 2D array, use a row position and a column position. The first number is the row; the second is the column:

```python
print(monthly_sales[0, 1])  # First row, second column
print(monthly_sales[0])     # First row
print(monthly_sales[:, 1])  # Second column from every row
```

The colon `:` means “take all” of that dimension. We can also select ranges of rows and columns. As with normal slicing, the ending position is not included:

```python
more_sales = np.array([[10, 20, 30, 40],
                       [50, 60, 70, 80],
                       [90, 100, 110, 120]])

print(more_sales[0:2, 1:3])  # First two rows, columns at positions 1 and 2
print(more_sales[:, 0:2])   # Every row, first two columns
print(more_sales[1:3, :])   # Last two rows, every column
```

Read `more_sales[0:2, 1:3]` as: “take rows 0 up to 2, and columns 1 up to 3.”

## A brief look at 3D and higher-dimensional arrays

A 3D array can be thought of as a stack of 2D tables. The example below has two tables, each with two rows and three columns:

```python
sales_by_month = np.array([
    [[10, 20, 30],
     [40, 50, 60]],
    [[15, 25, 35],
     [45, 55, 65]]
])

print(sales_by_month.shape)  # (2, 2, 3): 2 tables, 2 rows, 3 columns
print(sales_by_month[0])     # The first 2D table
print(sales_by_month[0, 1, 2])  # First table, second row, third column
```

Arrays can have more dimensions than three. These are called **N-dimensional** or **ND arrays**. You usually do not need to create high-dimensional arrays as a beginner; it is enough to know that NumPy can store data in more than one, two, or three dimensions. The `.ndim` property tells you how many dimensions an array has.

## 5. Select a range with slicing

Slicing selects part of an array. The ending position is not included.

```python
print(sales[1:3])  # Values at positions 1 and 2
print(sales[:2])   # From the start up to position 2
```

## 6. Do calculations

NumPy applies simple arithmetic to every value in the array:

```python
print(sales + 10)  # Add 10 to every value
print(sales * 2)   # Multiply every value by 2
print(sales / 2)   # Divide every value by 2
```

Two arrays of the same shape can also be calculated together, position by position:

```python
sales_week_1 = np.array([100, 150, 200])
sales_week_2 = np.array([120, 130, 180])

print(sales_week_1 + sales_week_2)
```

### Broadcasting: using one value with a whole array

Broadcasting means NumPy can use a single value in a calculation with every value in an array:

```python
prices = np.array([[10, 20],
                   [30, 40]])

prices_with_tax = prices + 2
print(prices_with_tax)
```

NumPy adds `2` to every value. This is a simple example of broadcasting.

## 7. Find basic summaries

```python
print("Total:", np.sum(sales))
print("Average:", np.mean(sales))
print("Smallest:", np.min(sales))
print("Largest:", np.max(sales))
```

For a 2D array, we can summarize all values or work by rows or columns:

```python
print("Total of all values:", np.sum(monthly_sales))
print("Total for each row:", np.sum(monthly_sales, axis=1))
print("Total for each column:", np.sum(monthly_sales, axis=0))
```

For this example, `axis=1` works across each row; `axis=0` works down each column.

## 8. Filter values

Use a condition inside square brackets to keep matching values:

```python
big_sales = sales[sales > 140]
print(big_sales)
```

The condition creates a True/False answer for each value, and NumPy returns the values where the answer is True.

Two conditions can be combined with `&` (and). Put each condition in parentheses:

```python
middle_sales = sales[(sales >= 120) & (sales <= 180)]
print(middle_sales)
```

## 9. Change the array shape

We can rearrange an array into rows and columns when the total number of values fits:

```python
numbers = np.array([1, 2, 3, 4, 5, 6])
table = numbers.reshape(2, 3)
print(table)
```

There are six values, so they fit in a 2-row, 3-column array.

## 10. A small practice example

```python
marks = np.array([65, 82, 49, 91, 73])

print("Average mark:", np.mean(marks))
print("Highest mark:", np.max(marks))
print("Marks at least 70:", marks[marks >= 70])
```

Try changing the values and the condition.

## Remember

- `np.array(...)` creates an array from values.
- Positions start at zero; `:` selects all values in one dimension.
- `.shape`, `.ndim`, `.size`, and `.dtype` describe an array.
- Arithmetic and summaries can be applied to arrays.
- A condition inside square brackets filters values.
- NumPy arrays are one of the foundations used by data analysis tools such as Pandas.
