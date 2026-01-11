**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# NumPy Lecture

---

## 1. NumPy Fundamentals

### Why NumPy for Data Analysis?

NumPy (Numerical Python) is the foundation of data analysis in Python because of two critical advantages:

**Speed and Memory Efficiency**
- NumPy arrays are stored in contiguous memory blocks
- Operations are implemented in C, making them 10-100x faster than pure Python
- Vectorized operations eliminate the need for explicit loops

```python
import numpy as np
import time

# Speed comparison: Python list vs NumPy array
python_list = list(range(1000000))
numpy_array = np.arange(1000000)

# Timing Python list operation
start = time.time()
result_list = [x * 2 for x in python_list]
python_time = time.time() - start

# Timing NumPy operation
start = time.time()
result_numpy = numpy_array * 2
numpy_time = time.time() - start

print(f"Python list time: {python_time:.4f} seconds")
print(f"NumPy array time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")
```

### Creating Arrays

**From Python Lists**
```python
# 1D array
data = [1, 2, 3, 4, 5]
arr1d = np.array(data)
print("1D Array:", arr1d)

# 2D array
data2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr2d = np.array(data2d)
print("2D Array:\n", arr2d)

# Mixed data types (will be converted to most general type)
mixed = [1, 2.5, 3]
arr_mixed = np.array(mixed)
print("Mixed array:", arr_mixed, "dtype:", arr_mixed.dtype)
```

**Loading from CSV Files**
```python
# Create sample CSV data first
sample_data = """1.2,2.3,3.4
4.5,5.6,6.7
7.8,8.9,9.0"""

with open('sample_data.csv', 'w') as f:
    f.write(sample_data)

# Load CSV data
csv_data = np.loadtxt('sample_data.csv', delimiter=',')
print("CSV Data:\n", csv_data)

# Load with specific data type
csv_int = np.loadtxt('sample_data.csv', delimiter=',', dtype=int)
print("CSV as integers:\n", csv_int)
```

**Array Creation Functions**
```python
# Zeros array
zeros_arr = np.zeros((3, 4))
print("Zeros array:\n", zeros_arr)

# Ones array
ones_arr = np.ones((2, 3))
print("Ones array:\n", ones_arr)

# Range arrays
range_arr = np.arange(0, 10, 2)  # start, stop, step
print("Range array:", range_arr)

# Evenly spaced values
linspace_arr = np.linspace(0, 1, 5)  # 5 values between 0 and 1
print("Linspace array:", linspace_arr)

# Random arrays
random_arr = np.random.random((2, 3))
print("Random array:\n", random_arr)
```

### Basic Array Properties

```python
sample_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print("Array:\n", sample_array)
print("Shape:", sample_array.shape)      # (rows, columns)
print("Data type:", sample_array.dtype)  # int64, float64, etc.
print("Size:", sample_array.size)        # total number of elements
print("Dimensions:", sample_array.ndim)  # number of dimensions
print("Item size:", sample_array.itemsize, "bytes")  # size of each element
```

---

## 2. Array Indexing and Selection

### Basic Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Single element access
print("First element:", arr[0])
print("Last element:", arr[-1])

# Slicing
print("First three:", arr[:3])
print("From index 2:", arr[2:])
print("Every other element:", arr[::2])

# 2D array indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array:\n", arr2d)
print("Element at [1,2]:", arr2d[1, 2])  # row 1, column 2
print("First row:", arr2d[0, :])         # all columns of first row
print("Second column:", arr2d[:, 1])     # all rows of second column

# Slicing 2D arrays
print("First 2 rows, first 2 columns:\n", arr2d[:2, :2])
```

### Boolean Indexing

Boolean indexing is one of NumPy's most powerful features for data filtering.

```python
data = np.array([1, 5, 3, 8, 2, 9, 4, 7, 6])

# Create boolean mask
mask = data > 5
print("Data:", data)
print("Mask (data > 5):", mask)
print("Filtered data:", data[mask])

# Direct boolean indexing
print("Values > 5:", data[data > 5])
print("Values between 3 and 7:", data[(data >= 3) & (data <= 7)])

# Multiple conditions
print("Values < 3 OR > 7:", data[(data < 3) | (data > 7)])

# 2D boolean indexing
sales_data = np.array([[100, 120, 80], 
                       [150, 90, 110], 
                       [130, 200, 95]])
print("Sales data:\n", sales_data)
print("Sales > 100:", sales_data[sales_data > 100])

# Finding positions of conditions
high_sales_positions = np.where(sales_data > 150)
print("Positions where sales > 150:", high_sales_positions)
```

---

## 3. Essential Operations

### Mathematical Operations

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

# Element-wise arithmetic
print("Addition:", arr1 + arr2)
print("Multiplication:", arr1 * arr2)
print("Division:", arr2 / arr1)
print("Power:", arr1 ** 2)

# Scalar operations
print("Add 5 to all:", arr1 + 5)
print("Multiply by 2:", arr1 * 2)

# Mathematical functions
print("Square root:", np.sqrt(arr1))
print("Exponential:", np.exp(arr1))
print("Sin values:", np.sin(arr1))
```

### Aggregation Functions

```python
sales_data = np.array([[100, 120, 80, 90], 
                       [150, 90, 110, 130], 
                       [130, 200, 95, 105]])

print("Sales data:\n", sales_data)

# Overall statistics
print("Sum of all sales:", np.sum(sales_data))
print("Mean sales:", np.mean(sales_data))
print("Min sales:", np.min(sales_data))
print("Max sales:", np.max(sales_data))
print("Standard deviation:", np.std(sales_data))

# Operations along axes
print("\nAxis operations:")
print("Sum by rows (axis=1):", np.sum(sales_data, axis=1))      # sum each row
print("Sum by columns (axis=0):", np.sum(sales_data, axis=0))   # sum each column
print("Mean by columns:", np.mean(sales_data, axis=0))
print("Max by rows:", np.max(sales_data, axis=1))
```

### Array Manipulation

```python
original = np.array([1, 2, 3, 4, 5, 6])

# Reshaping
reshaped = original.reshape(2, 3)
print("Original:", original)
print("Reshaped (2x3):\n", reshaped)

# Transpose
print("Transposed:\n", reshaped.T)

# Flattening
flattened = reshaped.flatten()
print("Flattened:", flattened)

# Concatenation
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate([arr1, arr2])
print("Concatenated:", concatenated)

# Horizontal and vertical stacking
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

h_stacked = np.hstack([matrix1, matrix2])
v_stacked = np.vstack([matrix1, matrix2])

print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Horizontal stack:\n", h_stacked)
print("Vertical stack:\n", v_stacked)
```

---

## 4. Data Cleaning Basics

### Handling Missing Data

```python
# Create data with NaN values
data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
print("Data with NaN:", data_with_nan)

# Find NaN values
nan_mask = np.isnan(data_with_nan)
print("NaN mask:", nan_mask)
print("Positions of NaN:", np.where(nan_mask))

# Remove NaN values
clean_data = data_with_nan[~nan_mask]  # ~ means NOT
print("Data without NaN:", clean_data)

# Replace NaN with specific value
data_filled = np.where(np.isnan(data_with_nan), 0, data_with_nan)
print("NaN replaced with 0:", data_filled)

# Replace with mean
mean_value = np.nanmean(data_with_nan)  # nanmean ignores NaN
data_mean_filled = np.where(np.isnan(data_with_nan), mean_value, data_with_nan)
print("NaN replaced with mean:", data_mean_filled)

# Working with 2D arrays containing NaN
sales_with_missing = np.array([[100, np.nan, 80], 
                               [150, 90, np.nan], 
                               [np.nan, 200, 95]])
print("Sales with missing:\n", sales_with_missing)
print("Mean ignoring NaN:", np.nanmean(sales_with_missing, axis=0))
```

### Data Type Conversion

```python
# String to numeric
string_numbers = np.array(['1', '2', '3', '4'])
numeric_array = string_numbers.astype(int)
print("String array:", string_numbers, "dtype:", string_numbers.dtype)
print("Numeric array:", numeric_array, "dtype:", numeric_array.dtype)

# Float to int
float_array = np.array([1.7, 2.3, 3.9])
int_array = float_array.astype(int)
print("Float array:", float_array)
print("Int array (truncated):", int_array)

# Boolean conversion
bool_array = np.array([0, 1, 2, 0, 3]).astype(bool)
print("Boolean array:", bool_array)

# Handling conversion errors safely
mixed_strings = np.array(['1', '2', 'abc', '4'])
try:
    converted = mixed_strings.astype(float)
except ValueError as e:
    print("Conversion error:", e)
    # Use pd.to_numeric with errors='coerce' in pandas for safer conversion
```

---

## 5. Useful Functions for Data Work

### Sorting and Searching

```python
unsorted_data = np.array([64, 34, 25, 12, 22, 11, 90])

# Sorting
sorted_data = np.sort(unsorted_data)
print("Original:", unsorted_data)
print("Sorted:", sorted_data)

# Get indices that would sort the array
sort_indices = np.argsort(unsorted_data)
print("Sort indices:", sort_indices)
print("Manually sorted:", unsorted_data[sort_indices])

# Sorting 2D arrays
scores = np.array([[85, 90, 78], 
                   [92, 88, 85], 
                   [78, 95, 82]])
print("Original scores:\n", scores)
print("Sorted by rows:\n", np.sort(scores, axis=1))
# Output: [[78 85 90]
#          [85 88 92]
#          [78 82 95]]

print("Sorted by columns:\n", np.sort(scores, axis=0))
# Output: [[78 88 78]
#          [85 90 82]
#          [92 95 85]]


```

### Unique Values

```python
data_with_duplicates = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])

# Find unique values
unique_values = np.unique(data_with_duplicates)
print("Original:", data_with_duplicates)
print("Unique values:", unique_values)

# Get unique values with counts
unique_vals, counts = np.unique(data_with_duplicates, return_counts=True)
print("Unique values:", unique_vals)
print("Counts:", counts)

# Working with 2D arrays
categories = np.array([['A', 'B', 'A'], 
                       ['C', 'B', 'A'], 
                       ['A', 'C', 'C']])
unique_categories = np.unique(categories)
print("Categories:\n", categories)
print("Unique categories:", unique_categories)
```

### Conditional Operations

```python
temperature_data = np.array([22, 25, 19, 30, 15, 28, 18])

# np.where for conditional selection
weather_description = np.where(temperature_data > 25, 'Hot', 'Moderate')
print("Temperatures:", temperature_data)
print("Weather:", weather_description)

# Multiple conditions with np.where
detailed_weather = np.where(temperature_data > 25, 'Hot',
                           np.where(temperature_data < 20, 'Cold', 'Moderate'))
print("Detailed weather:", detailed_weather)

# Conditional replacement
adjusted_temps = np.where(temperature_data < 20, temperature_data + 5, temperature_data)
print("Adjusted temperatures:", adjusted_temps)


```

---

## 6. Practical Example: Simple Data Analysis Workflow

Let's put everything together with a realistic example analyzing sales data.

```python
# Create sample sales data
np.random.seed(42)  # for reproducible results

# Generate sample data: 12 months, 5 products
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']

# Sales data (12 months x 5 products)
sales_data = np.random.randint(50, 200, size=(12, 5))

# Introduce some missing data
sales_data[2, 1] = np.nan  # March, Product B
sales_data[7, 3] = np.nan  # August, Product D

print("=== SALES DATA ANALYSIS ===")
print("Sales Data Shape:", sales_data.shape)
print("Sales Data (first 6 months):\n", sales_data[:6])

# 1. Basic Statistics
print("\n=== BASIC STATISTICS ===")
print("Total Sales (ignoring NaN):", np.nansum(sales_data))
print("Average Monthly Sales:", np.nanmean(sales_data))
print("Best Single Month-Product Sale:", np.nanmax(sales_data))
print("Worst Single Month-Product Sale:", np.nanmin(sales_data))

# 2. Handle Missing Data
print("\n=== MISSING DATA HANDLING ===")
missing_data_positions = np.where(np.isnan(sales_data))
print("Missing data at positions (month, product):", 
      list(zip(missing_data_positions[0], missing_data_positions[1])))

# Fill missing data with column (product) mean
for col in range(sales_data.shape[1]):
    col_mean = np.nanmean(sales_data[:, col])
    sales_data[:, col] = np.where(np.isnan(sales_data[:, col]), col_mean, sales_data[:, col])

print("Missing data filled with product averages")

# 3. Product Analysis
print("\n=== PRODUCT PERFORMANCE ===")
product_totals = np.sum(sales_data, axis=0)
product_averages = np.mean(sales_data, axis=0)

for i, product in enumerate(products):
    print(f"{product}: Total=${product_totals[i]:.0f}, Average=${product_averages[i]:.1f}")

best_product_idx = np.argmax(product_totals)
print(f"\nBest performing product: {products[best_product_idx]}")

# 4. Monthly Analysis
print("\n=== MONTHLY TRENDS ===")
monthly_totals = np.sum(sales_data, axis=1)
monthly_averages = np.mean(sales_data, axis=1)

# Find best and worst months
best_month_idx = np.argmax(monthly_totals)
worst_month_idx = np.argmin(monthly_totals)

print(f"Best month: {months[best_month_idx]} (${monthly_totals[best_month_idx]:.0f})")
print(f"Worst month: {months[worst_month_idx]} (${monthly_totals[worst_month_idx]:.0f})")

# 6. Growth Analysis (simple month-to-month)
print("\n=== GROWTH ANALYSIS ===")
monthly_growth = np.diff(monthly_totals)  # difference between consecutive months
positive_growth_months = monthly_growth > 0

print("Months with positive growth:", np.array(months[1:])[positive_growth_months])
print("Average monthly growth:", np.mean(monthly_growth))

# 7. Data Quality Check
print("\n=== DATA QUALITY SUMMARY ===")
print("Total data points:", sales_data.size)
print("Data range: ${:.0f} - ${:.0f}".format(np.min(sales_data), np.max(sales_data)))
print("Standard deviation: ${:.1f}".format(np.std(sales_data)))

# Identify outliers using the 2-sigma rule (standard statistical method)
# The 2-sigma rule states that ~95% of data should fall within 2 standard deviations
# of the mean in a normal distribution. Values beyond this are considered outliers.
mean_sales = np.mean(sales_data)
std_sales = np.std(sales_data)
outliers = np.abs(sales_data - mean_sales) > 2 * std_sales  # |x - μ| > 2σ
outlier_count = np.sum(outliers)
print("Potential outliers:", outlier_count, "out of", sales_data.size, "data points")

print("\n=== ANALYSIS COMPLETE ===")
```

---

## Key Takeaways

1. **NumPy is Essential**: Foundation for all data analysis in Python
2. **Vectorization**: Always prefer array operations over loops
3. **Boolean Indexing**: Powerful tool for data filtering
4. **Axis Parameter**: Critical for operations on multidimensional data
5. **Handle Missing Data**: Use `np.nan` functions for robust analysis
6. **Memory Efficiency**: NumPy arrays use significantly less memory than Python lists

## Next Steps

After mastering NumPy fundamentals:
- Learn **Pandas** for more advanced data manipulation
- Explore **Matplotlib** for data visualization  
- Study **Scipy** for statistical analysis
- Practice with real datasets to solidify these concepts

Remember: NumPy is the foundation that everything else builds upon in the Python data science ecosystem!

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*