**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*

---

# Data Visualization Lecture — Beginner Sales Examples

## Why make charts?

A chart helps us see patterns in data. We will use simple charts and the provided `data/sales.csv` file.

Run the examples from the main course folder.

## Three ways to study data

### 1. Univariate: one variable

“Uni” means one. We look at one column at a time. For example: How are the sales amounts spread out? A histogram can help answer this.

### 2. Bivariate: two variables

“Bi” means two. We compare two columns. For example: Do higher sales tend to come with higher profit? A scatter plot can help us see the relationship.

### 3. Multivariate: three or more variables

We look at three or more columns together. For example: compare sales and profit, use product type as the color, and use market as a group. Keep the chart readable by adding only a little extra information.

## Prepare the data and chart tools

```python
import pandas as pd
import matplotlib.pyplot as plt

sales = pd.read_csv("data/sales.csv")
```

Pandas reads the table. Matplotlib makes the charts.

## 1. Univariate: distribution of sales

```python
plt.hist(sales["Sales"])
plt.title("How sales amounts are spread out")
plt.xlabel("Sales")
plt.ylabel("Number of records")
plt.show()
```

Each bar shows how many records fall within a range of sales amounts. Ask students: Are most records at the low, middle, or high end?

## 2. Bivariate: sales and profit

```python
plt.scatter(sales["Sales"], sales["Profit"])
plt.title("Sales and Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()
```

Each dot is one record. The chart compares two numerical columns. Ask: Do the dots generally move upward as sales increase?

## 3. Multivariate: sales, profit, and product type

We can add a third variable by making one scatter plot for each product type. This keeps the code simple and the groups easy to compare.

```python
coffee = sales[sales["Product Type"] == "Coffee"]
tea = sales[sales["Product Type"] == "Tea"]

plt.scatter(coffee["Sales"], coffee["Profit"], label="Coffee")
plt.scatter(tea["Sales"], tea["Profit"], label="Tea")
plt.title("Sales and Profit by Product Type")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.legend()
plt.show()
```

The horizontal position represents Sales, the vertical position represents Profit, and the legend identifies Product Type. The dataset includes Coffee and Tea records.

## Optional multivariate example: add a market filter

This example keeps the same two measurements and product groups, but shows only one market. Change `East` to another market value from the file.

```python
east = sales[sales["Market"] == "East"]
east_coffee = east[east["Product Type"] == "Coffee"]
east_tea = east[east["Product Type"] == "Tea"]

plt.scatter(east_coffee["Sales"], east_coffee["Profit"], label="Coffee")
plt.scatter(east_tea["Sales"], east_tea["Profit"], label="Tea")
plt.title("Sales and Profit in the East Market")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.legend()
plt.show()
```

## A good order for making charts

1. Start with one column and get to know it.
2. Compare two columns to ask a relationship question.
3. Add another category only when it helps answer the question.
4. Give the chart a title and label both axes.
5. Look at the chart and describe what you see in plain language.

## Remember

- Univariate means one variable.
- Bivariate means two variables.
- Multivariate means three or more variables.
- A chart should help answer a question.
- Titles, axis labels, and a legend help everyone read the chart.
