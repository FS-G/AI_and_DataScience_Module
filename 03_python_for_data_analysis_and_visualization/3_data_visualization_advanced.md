**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*

---

# Data Visualization Lecture — Advanced

## What we will learn

We will explore the same three stages in order:

1. **Univariate:** study one variable.
2. **Bivariate:** compare two variables.
3. **Multivariate:** look at three or more variables together.

For each stage, we will use common chart types and see simple ways to make them with Matplotlib, Seaborn, and Plotly. Some charts are specific to one library; where a library has no suitable built-in chart, we will use the appropriate library rather than force a complicated example.

## 1. Prepare the data

We will use `data/visualization_sales.csv`, a sample dataset made for this lecture. It contains 600 orders and both numerical and categorical information.

Numerical columns include `Sales`, `COGS`, `Profit`, `Quantity`, `Discount`, and `Rating`. Categories include `Region`, `City`, `Category`, `Product`, `CustomerSegment`, and `Channel`.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("data/visualization_sales.csv")
data["Date"] = pd.to_datetime(data["Date"])

print(data.head())
```

If a library is not installed, install it in the terminal with:

```text
pip install pandas matplotlib seaborn plotly
```

Each example below can be run on its own after the imports and data-loading code above.

# Part 1: Univariate — one variable

Univariate charts help us understand one column: its typical values, spread, unusual values, or most common categories.

## 1.1 Numerical distribution: histogram

A histogram groups numbers into ranges. Here we look at order Sales.

**Matplotlib**

```python
plt.hist(data["Sales"], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Number of orders")
plt.show()
```

**Seaborn**

```python
sns.histplot(data=data, x="Sales", bins=20)
plt.title("Sales Distribution")
plt.show()
```

**Plotly**

```python
fig = px.histogram(data, x="Sales", nbins=20, title="Sales Distribution")
fig.show()
```

## 1.2 Numerical distribution: density curve

A density curve is a smoothed view of where values are concentrated.

**Matplotlib** (a density-scaled histogram)

```python
plt.hist(data["Sales"], bins=20, density=True)
plt.title("Sales Density")
plt.xlabel("Sales")
plt.ylabel("Density")
plt.show()
```

**Seaborn** (a smoothed KDE curve)

```python
sns.kdeplot(data=data, x="Sales", fill=True)
plt.title("Sales Density Curve")
plt.show()
```

**Plotly** (density-scaled histogram)

```python
fig = px.histogram(data, x="Sales", histnorm="probability density",
                   title="Sales Density")
fig.show()
```

## 1.3 Numerical distribution: box plot

A box plot gives a compact view of the middle of the data and can help spot unusually low or high values.

**Matplotlib**

```python
plt.boxplot(data["Sales"])
plt.title("Sales Box Plot")
plt.ylabel("Sales")
plt.show()
```

**Seaborn**

```python
sns.boxplot(data=data, y="Sales")
plt.title("Sales Box Plot")
plt.show()
```

**Plotly**

```python
fig = px.box(data, y="Sales", title="Sales Box Plot")
fig.show()
```

## 1.4 Numerical distribution: violin plot

A violin plot combines a box-like summary with a shape showing where values are concentrated.

**Matplotlib**

```python
plt.violinplot(data["Sales"])
plt.title("Sales Violin Plot")
plt.ylabel("Sales")
plt.show()
```

**Seaborn**

```python
sns.violinplot(data=data, y="Sales")
plt.title("Sales Violin Plot")
plt.show()
```

**Plotly**

```python
fig = px.violin(data, y="Sales", title="Sales Violin Plot")
fig.show()
```

## 1.5 Numerical distribution: ECDF

An empirical cumulative distribution function (ECDF) shows the share of orders with a value at or below each Sales amount.

**Matplotlib**

```python
sorted_sales = data["Sales"].sort_values()
share = range(1, len(sorted_sales) + 1)
plt.plot(sorted_sales, [value / len(sorted_sales) for value in share])
plt.title("Cumulative Share of Orders by Sales")
plt.xlabel("Sales")
plt.ylabel("Share at or below this value")
plt.show()
```

**Seaborn**

```python
sns.ecdfplot(data=data, x="Sales")
plt.title("Cumulative Share of Orders by Sales")
plt.show()
```

**Plotly**

```python
fig = px.ecdf(data, x="Sales", title="Cumulative Share of Orders by Sales")
fig.show()
```

## 1.6 Categorical counts: bar / count plot

A count plot shows how many records belong to each category. We will count orders by Product.

**Matplotlib**

```python
product_counts = data["Product"].value_counts()
plt.bar(product_counts.index, product_counts.values)
plt.title("Orders by Product")
plt.xlabel("Product")
plt.ylabel("Number of orders")
plt.show()
```

**Seaborn**

```python
sns.countplot(data=data, x="Product")
plt.title("Orders by Product")
plt.show()
```

**Plotly**

```python
product_counts = data["Product"].value_counts().rename_axis("Product").reset_index(name="Orders")
fig = px.bar(product_counts, x="Product", y="Orders", title="Orders by Product")
fig.show()
```

## 1.7 Categorical share: pie chart

A pie chart shows how a total is divided between categories. It is easiest to read when there are only a few categories.

**Matplotlib**

```python
category_counts = data["Category"].value_counts()
plt.pie(category_counts.values, labels=category_counts.index, autopct="%1.0f%%")
plt.title("Share of Orders by Category")
plt.show()
```

**Plotly**

```python
category_counts = data["Category"].value_counts().rename_axis("Category").reset_index(name="Orders")
fig = px.pie(category_counts, names="Category", values="Orders",
              title="Share of Orders by Category")
fig.show()
```

Seaborn does not have a built-in pie chart. Matplotlib or Plotly is the simple choice for this chart.

# Part 2: Bivariate — two variables

Bivariate charts help us compare two columns. We can ask whether two numerical values move together, whether a number changes over time, or whether a number differs across categories.

## 2.1 Numerical and numerical: scatter plot

Each dot is an order. The chart compares Sales and Profit.

**Matplotlib**

```python
plt.scatter(data["Sales"], data["Profit"])
plt.title("Sales and Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()
```

**Seaborn**

```python
sns.scatterplot(data=data, x="Sales", y="Profit")
plt.title("Sales and Profit")
plt.show()
```

**Plotly**

```python
fig = px.scatter(data, x="Sales", y="Profit", title="Sales and Profit")
fig.show()
```

## 2.2 Numerical and numerical: relationship line

Use a line chart when the horizontal value has a meaningful order, such as date. First, add Sales for each date:

```python
daily_sales = data.groupby("Date", as_index=False)["Sales"].sum()
```

**Matplotlib**

```python
plt.plot(daily_sales["Date"], daily_sales["Sales"])
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Seaborn**

```python
sns.lineplot(data=daily_sales, x="Date", y="Sales")
plt.title("Sales Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Plotly**

```python
fig = px.line(daily_sales, x="Date", y="Sales", title="Sales Over Time")
fig.show()
```

## 2.3 Numerical and numerical: density / hexbin chart

When many points overlap, a density chart shows where points are concentrated. A hexbin chart groups points into hexagonal cells.

**Matplotlib**

```python
plt.hexbin(data["Sales"], data["Profit"], gridsize=20)
plt.title("Sales and Profit: Point Density")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.colorbar(label="Number of orders")
plt.show()
```

**Seaborn**

```python
sns.kdeplot(data=data, x="Sales", y="Profit", fill=True)
plt.title("Sales and Profit Density")
plt.show()
```

**Plotly**

```python
fig = px.density_heatmap(data, x="Sales", y="Profit",
                         title="Sales and Profit: Point Density")
fig.show()
```

## 2.4 Category and number: compare averages with a bar chart

Here we compare average Sales across product categories. Bar charts are useful for comparing a number across groups.

```python
average_sales = data.groupby("Category", as_index=False)["Sales"].mean()
```

**Matplotlib**

```python
plt.bar(average_sales["Category"], average_sales["Sales"])
plt.title("Average Sales by Category")
plt.xlabel("Category")
plt.ylabel("Average sales")
plt.show()
```

**Seaborn**

```python
sns.barplot(data=data, x="Category", y="Sales")
plt.title("Average Sales by Category")
plt.show()
```

**Plotly**

```python
fig = px.bar(average_sales, x="Category", y="Sales", title="Average Sales by Category")
fig.show()
```

## 2.5 Category and number: box plot by group

A grouped box plot compares the spread of Sales for each product category.

**Matplotlib**

```python
category_names = data["Category"].unique()
groups = [data.loc[data["Category"] == category, "Sales"]
          for category in category_names]
plt.boxplot(groups)
plt.xticks(range(1, len(category_names) + 1), category_names)
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()
```

**Seaborn**

```python
sns.boxplot(data=data, x="Category", y="Sales")
plt.title("Sales by Category")
plt.show()
```

**Plotly**

```python
fig = px.box(data, x="Category", y="Sales", title="Sales by Category")
fig.show()
```

## 2.6 Category and number: violin plot by group

A grouped violin plot compares both the spread and shape of Sales across categories.

**Matplotlib**

```python
plt.violinplot(groups, showmeans=True)
plt.title("Sales by Category")
plt.xlabel("Category positions")
plt.ylabel("Sales")
plt.show()
```

**Seaborn**

```python
sns.violinplot(data=data, x="Category", y="Sales")
plt.title("Sales by Category")
plt.show()
```

**Plotly**

```python
fig = px.violin(data, x="Category", y="Sales", title="Sales by Category")
fig.show()
```

## 2.7 Category and category: grouped counts

A grouped bar chart compares order counts for two categories. We will compare Channel within each Region.

```python
channel_region = pd.crosstab(data["Region"], data["Channel"])
```

**Matplotlib**

```python
channel_region.plot(kind="bar")
plt.title("Orders by Region and Channel")
plt.xlabel("Region")
plt.ylabel("Orders")
plt.show()
```

**Seaborn** (count chart, with color separating the second category)

```python
sns.countplot(data=data, x="Region", hue="Channel")
plt.title("Orders by Region and Channel")
plt.show()
```

**Plotly**

```python
fig = px.bar(channel_region.reset_index(), x="Region", y=list(channel_region.columns),
             barmode="group", title="Orders by Region and Channel")
fig.show()
```

The same grouped counts can be shown as a **stacked bar chart**. In Matplotlib, use `channel_region.plot(kind="bar", stacked=True)`. In Plotly, set `barmode="stack"` instead of `barmode="group"` in the example above.

## 2.8 Category and category: heatmap of counts

A heatmap uses color to make high and low counts easy to compare.

**Matplotlib**

```python
plt.imshow(channel_region, aspect="auto")
plt.title("Orders by Region and Channel")
plt.xticks(range(len(channel_region.columns)), channel_region.columns)
plt.yticks(range(len(channel_region.index)), channel_region.index)
plt.colorbar(label="Orders")
plt.show()
```

**Seaborn**

```python
sns.heatmap(channel_region, annot=True, fmt="d")
plt.title("Orders by Region and Channel")
plt.show()
```

**Plotly**

```python
fig = px.imshow(channel_region, aspect="auto", title="Orders by Region and Channel")
fig.show()
```

# Part 3: Multivariate — three or more variables

Multivariate charts add information such as a color, size, panel, or extra axis. Add only details that help answer a question.

## 3.1 Colored scatter plot

Compare Sales and Profit, then use color to show Region. This displays three variables: Sales, Profit, and Region.

**Matplotlib**

```python
for region in data["Region"].unique():
    part = data[data["Region"] == region]
    plt.scatter(part["Sales"], part["Profit"], label=region)
plt.title("Sales and Profit by Region")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.legend()
plt.show()
```

**Seaborn**

```python
sns.scatterplot(data=data, x="Sales", y="Profit", hue="Region")
plt.title("Sales and Profit by Region")
plt.show()
```

**Plotly**

```python
fig = px.scatter(data, x="Sales", y="Profit", color="Region",
                 title="Sales and Profit by Region")
fig.show()
```

## 3.2 Bubble chart

A bubble chart adds a size variable. Here, bubble size represents Quantity and color represents Category.

**Matplotlib**

```python
plt.scatter(data["Sales"], data["Profit"], s=data["Quantity"] * 12,
            alpha=0.5)
plt.title("Sales and Profit (Bubble Size = Quantity)")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()
```

**Seaborn**

```python
sns.scatterplot(data=data, x="Sales", y="Profit",
                size="Quantity", hue="Category", sizes=(20, 200))
plt.title("Sales and Profit by Quantity and Category")
plt.show()
```

**Plotly**

```python
fig = px.scatter(data, x="Sales", y="Profit", size="Quantity", color="Category",
                 title="Sales and Profit by Quantity and Category")
fig.show()
```

## 3.3 Scatter plot matrix / pair plot

A scatter plot matrix shows pairwise relationships between several numerical columns. Use a small sample to keep it readable.

**Matplotlib (through Pandas)**

```python
small_data = data[["Sales", "Profit", "Quantity", "Discount"]].head(100)
pd.plotting.scatter_matrix(small_data, figsize=(8, 8))
plt.show()
```

**Seaborn**

```python
sns.pairplot(data=data.head(100), vars=["Sales", "Profit", "Quantity", "Discount"])
plt.show()
```

**Plotly**

```python
fig = px.scatter_matrix(data.head(100),
                        dimensions=["Sales", "Profit", "Quantity", "Discount"],
                        title="Relationships Between Sales Measures")
fig.show()
```

## 3.4 Correlation heatmap

Correlation summarizes how numerical columns move together. Values closer to 1 move together, values near -1 move in opposite directions, and values near 0 have little straight-line relationship.

```python
correlation = data[["Sales", "COGS", "Profit", "Quantity", "Discount", "Rating"]].corr()
```

**Matplotlib**

```python
plt.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.colorbar()
plt.title("Correlation Between Numerical Columns")
plt.tight_layout()
plt.show()
```

**Seaborn**

```python
sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Numerical Columns")
plt.show()
```

**Plotly**

```python
fig = px.imshow(correlation, color_continuous_scale="RdBu",
                 title="Correlation Between Numerical Columns")
fig.show()
```

## 3.5 Faceted charts

Faceting creates a small chart for each group. Here we compare Sales and Profit separately for each Channel.

**Matplotlib**

```python
channels = data["Channel"].unique()
fig, axes = plt.subplots(1, len(channels), figsize=(10, 4))
for i, channel in enumerate(channels):
    part = data[data["Channel"] == channel]
    axes[i].scatter(part["Sales"], part["Profit"])
    axes[i].set_title(channel)
    axes[i].set_xlabel("Sales")
    axes[i].set_ylabel("Profit")
plt.tight_layout()
plt.show()
```

**Seaborn**

```python
grid = sns.relplot(data=data, x="Sales", y="Profit", col="Channel")
grid.set_axis_labels("Sales", "Profit")
plt.show()
```

**Plotly**

```python
fig = px.scatter(data, x="Sales", y="Profit", facet_col="Channel",
                 title="Sales and Profit by Channel")
fig.show()
```

## 3.6 3D scatter plot

A 3D scatter plot places one numerical variable on each axis. Color can show a fourth category.

**Matplotlib**

```python
fig = plt.figure()
axis = fig.add_subplot(111, projection="3d")
axis.scatter(data["Sales"], data["Profit"], data["Quantity"])
axis.set_xlabel("Sales")
axis.set_ylabel("Profit")
axis.set_zlabel("Quantity")
plt.show()
```

**Plotly**

```python
fig = px.scatter_3d(data, x="Sales", y="Profit", z="Quantity", color="Region",
                    title="Sales, Profit, and Quantity by Region")
fig.show()
```

Seaborn does not provide a built-in 3D scatter chart. Matplotlib and Plotly are suitable choices.

## 3.7 Parallel coordinates

Parallel coordinates show several numerical values for each order as a line across a set of axes. This chart is most straightforward in Plotly.

```python
fig = px.parallel_coordinates(
    data.head(100),
    dimensions=["Sales", "Profit", "Quantity", "Discount", "Rating"],
    color="Rating",
    title="Order Measures in Parallel Coordinates"
)
fig.show()
```

Plotly provides a direct parallel-coordinates chart. Matplotlib and Seaborn do not have a similarly simple built-in version for beginners.

## 3.8 Parallel categories

Parallel categories show how categorical groups connect, such as which Regions, Categories, Channels, and Customer Segments appear together. Plotly provides a direct chart for this:

```python
fig = px.parallel_categories(
    data.head(100),
    dimensions=["Region", "Category", "Channel", "CustomerSegment"],
    title="Orders Across Categorical Groups"
)
fig.show()
```

This chart is best used with a smaller number of rows so the paths remain readable.

## 3.9 Sunburst chart

A sunburst chart shows a hierarchy as nested rings. This example shows Region, then Category, then Product, with Sales represented by segment size.

```python
fig = px.sunburst(data, path=["Region", "Category", "Product"], values="Sales",
                  title="Sales by Region, Category, and Product")
fig.show()
```

Plotly Express has a direct sunburst chart. Matplotlib and Seaborn do not have an equally simple built-in version.

## 3.10 Treemap

A treemap uses nested rectangles to show the same kind of hierarchy. Larger rectangles represent more Sales.

```python
fig = px.treemap(data, path=["Region", "Category", "Product"], values="Sales",
                 title="Sales by Region, Category, and Product")
fig.show()
```

Plotly Express has a direct treemap chart. Matplotlib and Seaborn do not have an equally simple built-in version.

## How to choose and explain a chart

1. Ask a clear question about the data.
2. Decide whether it uses one, two, or several variables.
3. Choose a chart that fits the variable types.
4. Add a clear title and labels.
5. Describe the pattern in plain language, without claiming more than the chart shows.

## Remember

- **Univariate:** one variable; use distributions or counts.
- **Bivariate:** two variables; compare relationships, groups, or counts.
- **Multivariate:** three or more variables; add color, size, panels, or axes carefully.
- Matplotlib, Seaborn, and Plotly can make many of the same chart types, but some specialized charts are best supported by one library.
- A clear question and readable chart matter more than adding many visual details.

## Library documentation

- [Matplotlib chart gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn user guide](https://seaborn.pydata.org/tutorial.html)
- [Plotly Express chart API](https://plotly.com/python/plotly-express/)
