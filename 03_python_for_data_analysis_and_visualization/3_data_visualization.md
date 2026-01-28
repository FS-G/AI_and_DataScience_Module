**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*

# Data Visualization with Python
## From Simple Charts to Interactive Dashboards

---

## Introduction: The Power of Visual Storytelling

Imagine you're a business analyst looking at a spreadsheet with thousands of sales records. Numbers, dates, categories—all staring back at you in rows and columns. Now imagine transforming that data into beautiful, insightful visualizations that tell a story at a glance. That's the power of data visualization.

In this lecture, we'll journey through three levels of visualization complexity:
1. **Univariate Visualization** - Understanding single variables
2. **Bivariate Visualization** - Exploring relationships between two variables
3. **Multivariate Visualization** - Uncovering patterns across multiple dimensions

We'll use three powerful libraries—**Matplotlib**, **Seaborn**, and **Plotly**—each bringing unique strengths to our visualization toolkit. And throughout this journey, we'll work with real sales data, discovering insights that numbers alone could never reveal.

Let's begin our visual storytelling adventure!

---

## Part 1: Setting Up Our Visualization Environment

Before we dive into creating visualizations, let's set up our workspace and load our data.

```python
# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')  # Modern, clean style
sns.set_palette("husl")  # Beautiful color palette
sns.set_style("whitegrid")

# Configure plotly to work offline (no internet needed)
import plotly.io as pio
pio.renderers.default = "notebook"  # For Jupyter notebooks
# For other environments, use: pio.renderers.default = "browser"

# Load our sales data
df = pd.read_csv('data/sales.csv')

# Clean column names (remove spaces)
df.columns = df.columns.str.replace(' ', '_')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y %H:%M:%S')

# Quick data exploration
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nBasic statistics:")
print(df[['Sales', 'Profit', 'COGS', 'Marketing']].describe())
```

---

## Part 2: Univariate Data Visualization
### Understanding Single Variables

Univariate visualization is where every data scientist begins. It's about understanding the distribution, central tendency, and spread of a single variable. Think of it as getting to know one character in our data story before introducing others.

### 2.1 Histograms: The Distribution Story

**Histograms** reveal how our data is distributed—are sales clustered around certain values? Do we have outliers? Let's explore sales distribution using all three libraries.

#### Matplotlib: The Foundation

```python
# Basic histogram with matplotlib
plt.figure(figsize=(10, 6))
plt.hist(df['Sales'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Sales ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Sales - Matplotlib', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Enhanced histogram with multiple elements
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with density curve
axes[0].hist(df['Sales'], bins=30, density=True, color='steelblue', 
             edgecolor='black', alpha=0.7, label='Sales Distribution')
axes[0].axvline(df['Sales'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: ${df["Sales"].mean():.2f}')
axes[0].axvline(df['Sales'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: ${df["Sales"].median():.2f}')
axes[0].set_xlabel('Sales ($)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Sales Distribution with Statistics', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative distribution
axes[1].hist(df['Sales'], bins=30, cumulative=True, color='coral', 
             edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Sales ($)', fontsize=11)
axes[1].set_ylabel('Cumulative Frequency', fontsize=11)
axes[1].set_title('Cumulative Sales Distribution', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Seaborn: Beautiful by Default

```python
# Seaborn makes beautiful histograms effortlessly
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Basic histogram
sns.histplot(data=df, x='Sales', bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Sales Distribution with KDE', fontsize=12, fontweight='bold')

# Histogram by Product Type
sns.histplot(data=df, x='Sales', hue='Product_Type', bins=30, 
             kde=True, alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title('Sales Distribution by Product Type', fontsize=12, fontweight='bold')
axes[0, 1].legend(title='Product Type')

# Multiple variables comparison
sns.histplot(data=df, x='Profit', bins=30, kde=True, ax=axes[1, 0], color='green')
axes[1, 0].set_title('Profit Distribution', fontsize=12, fontweight='bold')

# Marketing expenses
sns.histplot(data=df, x='Marketing', bins=30, kde=True, ax=axes[1, 1], color='orange')
axes[1, 1].set_title('Marketing Expenses Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

#### Plotly: Interactive Exploration

```python
# Plotly histogram - interactive and engaging
fig = px.histogram(df, x='Sales', nbins=30,
                   title='Interactive Sales Distribution',
                   labels={'Sales': 'Sales ($)', 'count': 'Frequency'},
                   color_discrete_sequence=['steelblue'])

# Add mean and median lines
mean_sales = df['Sales'].mean()
median_sales = df['Sales'].median()

fig.add_vline(x=mean_sales, line_dash="dash", line_color="red",
              annotation_text=f"Mean: ${mean_sales:.2f}")
fig.add_vline(x=median_sales, line_dash="dash", line_color="green",
              annotation_text=f"Median: ${median_sales:.2f}")

# Update layout for better appearance
fig.update_layout(
    xaxis_title="Sales ($)",
    yaxis_title="Frequency",
    title_font_size=16,
    showlegend=False,
    template='plotly_white'
)

fig.show()

# Interactive histogram with hover information
fig = px.histogram(df, x='Sales', nbins=30,
                   title='Sales Distribution - Hover for Details',
                   labels={'Sales': 'Sales ($)', 'count': 'Frequency'},
                   hover_data=['Profit', 'Product_Type'],
                   color_discrete_sequence=['steelblue'])

fig.update_traces(hovertemplate='<b>Sales:</b> $%{x:.2f}<br>' +
                                 '<b>Frequency:</b> %{y}<br>' +
                                 '<b>Product Type:</b> %{customdata[1]}<extra></extra>')

fig.update_layout(template='plotly_white')
fig.show()
```

### 2.2 Box Plots: The Outlier Detectives

**Box plots** are excellent for identifying outliers and understanding the spread of data. They show quartiles, median, and potential outliers all in one view.

```python
# Matplotlib box plot
plt.figure(figsize=(10, 6))
plt.boxplot(df['Sales'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
plt.ylabel('Sales ($)', fontsize=12)
plt.title('Sales Distribution - Box Plot (Matplotlib)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# Seaborn box plot - more informative
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Single variable
sns.boxplot(y=df['Sales'], ax=axes[0], color='steelblue')
axes[0].set_ylabel('Sales ($)', fontsize=12)
axes[0].set_title('Sales Distribution', fontsize=12, fontweight='bold')

# By Product Type
sns.boxplot(data=df, x='Product_Type', y='Sales', ax=axes[1])
axes[1].set_xlabel('Product Type', fontsize=12)
axes[1].set_ylabel('Sales ($)', fontsize=12)
axes[1].set_title('Sales by Product Type', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Plotly box plot - interactive
fig = px.box(df, y='Sales', title='Interactive Sales Box Plot',
             labels={'Sales': 'Sales ($)'})
fig.update_layout(template='plotly_white')
fig.show()

# Box plot by Product Type with Plotly
fig = px.box(df, x='Product_Type', y='Sales',
             title='Sales Distribution by Product Type',
             labels={'Sales': 'Sales ($)', 'Product_Type': 'Product Type'},
             color='Product_Type')
fig.update_layout(template='plotly_white', showlegend=False)
fig.show()
```

### 2.3 Violin Plots: Distribution Shape Revealed

**Violin plots** combine the benefits of box plots and density plots, showing both summary statistics and distribution shape.

```python
# Seaborn violin plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Single variable
sns.violinplot(y=df['Sales'], ax=axes[0], color='steelblue')
axes[0].set_ylabel('Sales ($)', fontsize=12)
axes[0].set_title('Sales Distribution - Violin Plot', fontsize=12, fontweight='bold')

# By Product Type
sns.violinplot(data=df, x='Product_Type', y='Sales', ax=axes[1])
axes[1].set_xlabel('Product Type', fontsize=12)
axes[1].set_ylabel('Sales ($)', fontsize=12)
axes[1].set_title('Sales Distribution by Product Type', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Plotly violin plot
fig = px.violin(df, y='Sales', title='Sales Distribution - Violin Plot',
                labels={'Sales': 'Sales ($)'})
fig.update_layout(template='plotly_white')
fig.show()
```

### 2.4 Bar Charts: Categorical Insights

When dealing with categorical data, **bar charts** are our go-to visualization.

```python
# Count of records by Product Type
product_counts = df['Product_Type'].value_counts()

# Matplotlib bar chart
plt.figure(figsize=(10, 6))
plt.bar(product_counts.index, product_counts.values, 
        color=['steelblue', 'coral', 'lightgreen', 'gold', 'plum'])
plt.xlabel('Product Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Number of Records by Product Type', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Seaborn bar chart
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Product_Type', palette='husl')
plt.xlabel('Product Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Product Type Distribution', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotly bar chart - interactive
fig = px.bar(x=product_counts.index, y=product_counts.values,
             title='Product Type Distribution',
             labels={'x': 'Product Type', 'y': 'Count'},
             color=product_counts.values,
             color_continuous_scale='Blues')
fig.update_layout(template='plotly_white', showlegend=False)
fig.update_xaxes(tickangle=45)
fig.show()

# Horizontal bar chart
fig = px.bar(x=product_counts.values, y=product_counts.index,
             orientation='h',
             title='Product Type Distribution (Horizontal)',
             labels={'x': 'Count', 'y': 'Product Type'},
             color=product_counts.values,
             color_continuous_scale='Viridis')
fig.update_layout(template='plotly_white', showlegend=False)
fig.show()
```

### 2.5 Pie Charts: Proportional Relationships

**Pie charts** show proportions, though they should be used sparingly (typically for 3-5 categories).

```python
# Calculate total sales by Product Type
sales_by_product = df.groupby('Product_Type')['Sales'].sum().sort_values(ascending=False)

# Matplotlib pie chart
plt.figure(figsize=(10, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
plt.pie(sales_by_product.values, labels=sales_by_product.index, 
        autopct='%1.1f%%', startangle=90, colors=colors,
        explode=(0.05, 0, 0, 0, 0))  # Explode the largest slice
plt.title('Total Sales by Product Type', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Plotly pie chart - interactive
fig = px.pie(values=sales_by_product.values, 
             names=sales_by_product.index,
             title='Total Sales by Product Type',
             color_discrete_sequence=px.colors.qualitative.Set3)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(template='plotly_white')
fig.show()
```

---

## Part 3: Bivariate Data Visualization
### Exploring Relationships Between Two Variables

Now that we understand individual variables, let's explore how they relate to each other. Bivariate visualization helps us answer questions like: "Do higher sales lead to higher profits?" or "Is there a relationship between marketing spend and sales?"

### 3.1 Scatter Plots: The Relationship Revealers

**Scatter plots** are perfect for exploring relationships between two continuous variables.

```python
# Matplotlib scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Sales'], df['Profit'], alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
plt.xlabel('Sales ($)', fontsize=12)
plt.ylabel('Profit ($)', fontsize=12)
plt.title('Sales vs Profit Relationship', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Enhanced scatter with regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['Sales'], df['Profit'], alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)

# Add regression line
z = np.polyfit(df['Sales'], df['Profit'], 1)
p = np.poly1d(z)
plt.plot(df['Sales'], p(df['Sales']), "r--", linewidth=2, label=f'Trend Line: y={z[0]:.2f}x+{z[1]:.2f}')

plt.xlabel('Sales ($)', fontsize=12)
plt.ylabel('Profit ($)', fontsize=12)
plt.title('Sales vs Profit with Trend Line', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Seaborn scatter plot with regression
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Sales', y='Profit', hue='Product_Type', 
                s=100, alpha=0.7, palette='husl')
sns.regplot(data=df, x='Sales', y='Profit', scatter=False, 
            color='red', line_kws={'linewidth': 2, 'linestyle': '--'})
plt.xlabel('Sales ($)', fontsize=12)
plt.ylabel('Profit ($)', fontsize=12)
plt.title('Sales vs Profit by Product Type', fontsize=14, fontweight='bold')
plt.legend(title='Product Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plotly scatter plot - highly interactive
fig = px.scatter(df, x='Sales', y='Profit',
                 title='Sales vs Profit - Interactive Exploration',
                 labels={'Sales': 'Sales ($)', 'Profit': 'Profit ($)'},
                 color='Product_Type',
                 size='Marketing',
                 hover_data=['COGS', 'Date'],
                 trendline='ols')  # Ordinary Least Squares regression

fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.update_layout(
    template='plotly_white',
    xaxis_title="Sales ($)",
    yaxis_title="Profit ($)",
    legend_title="Product Type"
)

fig.show()
```

### 3.2 Line Charts: Time Series Stories

When one variable is time, **line charts** tell the temporal story of our data.

```python
# Prepare time series data
df_time = df.groupby('Date')['Sales'].sum().reset_index()
df_time = df_time.sort_values('Date')

# Matplotlib line chart
plt.figure(figsize=(14, 6))
plt.plot(df_time['Date'], df_time['Sales'], marker='o', linewidth=2, 
         markersize=4, color='steelblue')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.title('Sales Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Multiple lines - Sales and Profit over time
df_time_multi = df.groupby('Date').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
df_time_multi = df_time_multi.sort_values('Date')

fig, ax1 = plt.subplots(figsize=(14, 6))

color1 = 'steelblue'
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Sales ($)', color=color1, fontsize=12)
line1 = ax1.plot(df_time_multi['Date'], df_time_multi['Sales'], 
                 color=color1, marker='o', linewidth=2, label='Sales')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = 'coral'
ax2.set_ylabel('Profit ($)', color=color2, fontsize=12)
line2 = ax2.plot(df_time_multi['Date'], df_time_multi['Profit'], 
                  color=color2, marker='s', linewidth=2, label='Profit')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Sales and Profit Over Time', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

# Plotly line chart - interactive time series
df_time_plotly = df.groupby(['Date', 'Product_Type'])['Sales'].sum().reset_index()
df_time_plotly = df_time_plotly.sort_values('Date')

fig = px.line(df_time_plotly, x='Date', y='Sales', color='Product_Type',
              title='Sales Over Time by Product Type',
              labels={'Sales': 'Sales ($)', 'Date': 'Date'},
              markers=True)

fig.update_layout(
    template='plotly_white',
    xaxis_title="Date",
    yaxis_title="Sales ($)",
    legend_title="Product Type",
    hovermode='x unified'
)

fig.show()
```

### 3.3 Heatmaps: Correlation Discovery

**Heatmaps** are excellent for visualizing correlation matrices and understanding relationships between multiple variables simultaneously.

```python
# Calculate correlation matrix
numeric_cols = ['Sales', 'Profit', 'COGS', 'Marketing', 'Total_Expenses']
corr_matrix = df[numeric_cols].corr()

# Matplotlib heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Heatmap - Matplotlib', fontsize=14, fontweight='bold')

# Add correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# Seaborn heatmap - much easier!
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Seaborn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Plotly heatmap - interactive
fig = px.imshow(corr_matrix,
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu',
                aspect="auto",
                title='Interactive Correlation Heatmap')

fig.update_layout(template='plotly_white')
fig.show()
```

### 3.4 Bar Charts: Comparing Categories

Grouped and stacked bar charts help us compare categories across different dimensions.

```python
# Sales by Product Type and Market
sales_by_product_market = df.groupby(['Product_Type', 'Market'])['Sales'].sum().reset_index()

# Seaborn grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=sales_by_product_market, x='Product_Type', y='Sales', 
            hue='Market', palette='husl')
plt.xlabel('Product Type', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.title('Sales by Product Type and Market', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(title='Market')
plt.tight_layout()
plt.show()

# Plotly grouped bar chart
fig = px.bar(sales_by_product_market, x='Product_Type', y='Sales',
             color='Market',
             title='Sales by Product Type and Market',
             labels={'Sales': 'Total Sales ($)', 'Product_Type': 'Product Type'},
             barmode='group')

fig.update_layout(template='plotly_white', xaxis_tickangle=-45)
fig.show()

# Stacked bar chart
fig = px.bar(sales_by_product_market, x='Product_Type', y='Sales',
             color='Market',
             title='Sales by Product Type and Market (Stacked)',
             labels={'Sales': 'Total Sales ($)', 'Product_Type': 'Product Type'},
             barmode='stack')

fig.update_layout(template='plotly_white', xaxis_tickangle=-45)
fig.show()
```

---

## Part 4: Multivariate Data Visualization
### Uncovering Complex Patterns

Multivariate visualization is where the magic happens. We can now explore how three or more variables interact, revealing insights that simpler visualizations might miss.

### 4.1 Scatter Plot Matrices: The Big Picture

**Scatter plot matrices** (pair plots) show relationships between multiple variables simultaneously.

```python
# Seaborn pair plot
# Select key numeric variables
pair_vars = ['Sales', 'Profit', 'COGS', 'Marketing']
pair_df = df[pair_vars + ['Product_Type']].dropna()

# Create pair plot
sns.pairplot(pair_df, hue='Product_Type', diag_kind='kde', 
             palette='husl', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot: Relationships Between Key Variables', 
             y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Plotly scatter matrix
fig = px.scatter_matrix(df, dimensions=['Sales', 'Profit', 'COGS', 'Marketing'],
                        color='Product_Type',
                        title='Scatter Matrix: Multivariate Relationships',
                        labels={col: col.replace('_', ' ') for col in ['Sales', 'Profit', 'COGS', 'Marketing']})

fig.update_traces(diagonal_visible=False, showupperhalf=False)
fig.update_layout(template='plotly_white', height=800)
fig.show()
```

### 4.2 3D Scatter Plots: Adding Depth

**3D scatter plots** allow us to visualize three continuous variables simultaneously.

```python
# Plotly 3D scatter plot
fig = px.scatter_3d(df, x='Sales', y='Profit', z='Marketing',
                    color='Product_Type',
                    size='COGS',
                    title='3D Visualization: Sales, Profit, and Marketing',
                    labels={'Sales': 'Sales ($)', 
                           'Profit': 'Profit ($)', 
                           'Marketing': 'Marketing ($)'},
                    hover_data=['Date', 'Market'])

fig.update_layout(template='plotly_white', scene=dict(
    xaxis_title='Sales ($)',
    yaxis_title='Profit ($)',
    zaxis_title='Marketing ($)'
))

fig.show()
```

### 4.3 Faceted Plots: Multiple Views

**Faceted plots** create multiple subplots, each showing a different subset of data.

```python
# Seaborn faceted scatter plots
g = sns.FacetGrid(df, col='Product_Type', col_wrap=3, height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x='Sales', y='Profit', hue='Market', alpha=0.7)
g.add_legend()
g.set_axis_labels('Sales ($)', 'Profit ($)')
plt.suptitle('Sales vs Profit by Product Type and Market', 
             y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Plotly faceted scatter plots
fig = px.scatter(df, x='Sales', y='Profit', color='Market',
                 facet_col='Product_Type', facet_col_wrap=3,
                 title='Sales vs Profit: Faceted by Product Type',
                 labels={'Sales': 'Sales ($)', 'Profit': 'Profit ($)'},
                 hover_data=['Date', 'COGS'])

fig.update_layout(template='plotly_white', height=600)
fig.show()

# Faceted line charts over time
df_time_facet = df.groupby(['Date', 'Product_Type'])['Sales'].sum().reset_index()
df_time_facet = df_time_facet.sort_values('Date')

fig = px.line(df_time_facet, x='Date', y='Sales', 
              facet_col='Product_Type', facet_col_wrap=3,
              title='Sales Over Time: Faceted by Product Type',
              labels={'Sales': 'Sales ($)', 'Date': 'Date'},
              markers=True)

fig.update_layout(template='plotly_white', height=600)
fig.show()
```

### 4.4 Bubble Charts: Size as a Dimension

**Bubble charts** add a fourth dimension (size) to scatter plots, making them powerful multivariate tools.

```python
# Aggregate data for bubble chart
bubble_data = df.groupby('Product_Type').agg({
    'Sales': 'mean',
    'Profit': 'mean',
    'Marketing': 'mean',
    'COGS': 'count'  # Count as size
}).reset_index()
bubble_data.columns = ['Product_Type', 'Avg_Sales', 'Avg_Profit', 'Avg_Marketing', 'Count']

# Plotly bubble chart
fig = px.scatter(bubble_data, x='Avg_Sales', y='Avg_Profit',
                 size='Count', color='Product_Type',
                 hover_name='Product_Type',
                 size_max=60,
                 title='Bubble Chart: Average Sales vs Profit (Size = Record Count)',
                 labels={'Avg_Sales': 'Average Sales ($)', 
                        'Avg_Profit': 'Average Profit ($)',
                        'Count': 'Number of Records'},
                 hover_data=['Avg_Marketing'])

fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
fig.update_layout(template='plotly_white')
fig.show()

# More complex bubble chart with multiple dimensions
fig = px.scatter(df, x='Sales', y='Profit',
                 size='Marketing', color='Product_Type',
                 facet_col='Market',
                 hover_data=['COGS', 'Date'],
                 title='Multivariate Bubble Chart: Sales, Profit, Marketing by Market',
                 labels={'Sales': 'Sales ($)', 
                        'Profit': 'Profit ($)',
                        'Marketing': 'Marketing ($)'},
                 size_max=30)

fig.update_layout(template='plotly_white', height=500)
fig.show()
```

### 4.5 Parallel Coordinates: Multidimensional Patterns

**Parallel coordinates plots** help visualize high-dimensional data by showing all variables simultaneously.

```python
# Sample data for parallel coordinates (too many points can be cluttered)
sample_df = df.sample(min(100, len(df)))[['Sales', 'Profit', 'COGS', 'Marketing', 'Product_Type']]

# Plotly parallel coordinates
fig = px.parallel_coordinates(sample_df,
                              color='Sales',
                              dimensions=['Sales', 'Profit', 'COGS', 'Marketing'],
                              labels={'Sales': 'Sales ($)',
                                     'Profit': 'Profit ($)',
                                     'COGS': 'COGS ($)',
                                     'Marketing': 'Marketing ($)'},
                              title='Parallel Coordinates: Multidimensional Data View',
                              color_continuous_scale=px.colors.sequential.Viridis)

fig.update_layout(template='plotly_white', height=600)
fig.show()

# Parallel categories for categorical data
fig = px.parallel_categories(df, dimensions=['Product_Type', 'Market', 'Market_Size'],
                            color='Sales',
                            title='Parallel Categories: Categorical Relationships',
                            color_continuous_scale=px.colors.sequential.Viridis)

fig.update_layout(template='plotly_white', height=600)
fig.show()
```

### 4.6 Sunburst Charts: Hierarchical Multivariate Views

**Sunburst charts** visualize hierarchical data with multiple levels.

```python
# Create hierarchical data
hierarchical_data = df.groupby(['Market', 'Product_Type'])['Sales'].sum().reset_index()

# Plotly sunburst chart
fig = px.sunburst(df, path=['Market', 'Product_Type'], values='Sales',
                  title='Sunburst Chart: Sales Hierarchy by Market and Product Type',
                  color='Sales',
                  color_continuous_scale='Viridis',
                  hover_data=['Profit'])

fig.update_layout(template='plotly_white')
fig.show()
```

### 4.7 Treemaps: Proportional Hierarchies

**Treemaps** show hierarchical data as nested rectangles, with size representing value.

```python
# Plotly treemap
fig = px.treemap(df, path=['Market', 'Product_Type'], values='Sales',
                 color='Profit',
                 title='Treemap: Sales by Market and Product Type (Color = Profit)',
                 color_continuous_scale='RdYlGn',
                 hover_data=['COGS', 'Marketing'])

fig.update_layout(template='plotly_white')
fig.show()
```

---

## Part 5: Advanced Visualization Techniques
### Combining Multiple Visualizations

### 5.1 Subplots: Multiple Views Together

```python
# Matplotlib subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top left: Sales distribution
axes[0, 0].hist(df['Sales'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Sales Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Sales ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Top right: Sales vs Profit scatter
axes[0, 1].scatter(df['Sales'], df['Profit'], alpha=0.6, s=30, c='coral')
axes[0, 1].set_title('Sales vs Profit', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Sales ($)')
axes[0, 1].set_ylabel('Profit ($)')
axes[0, 1].grid(True, alpha=0.3)

# Bottom left: Sales by Product Type
product_sales = df.groupby('Product_Type')['Sales'].sum()
axes[1, 0].bar(product_sales.index, product_sales.values, color='lightgreen')
axes[1, 0].set_title('Total Sales by Product Type', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Product Type')
axes[1, 0].set_ylabel('Total Sales ($)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Bottom right: Time series
df_time = df.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
axes[1, 1].plot(df_time['Date'], df_time['Sales'], marker='o', linewidth=2, color='purple')
axes[1, 1].set_title('Sales Over Time', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Total Sales ($)')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Comprehensive Sales Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# Plotly subplots
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sales Distribution', 'Sales vs Profit', 
                    'Sales by Product Type', 'Sales Over Time'),
    specs=[[{"type": "histogram"}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "scatter"}]]
)

# Sales distribution
fig.add_trace(
    go.Histogram(x=df['Sales'], nbinsx=30, name='Sales', marker_color='steelblue'),
    row=1, col=1
)

# Sales vs Profit
fig.add_trace(
    go.Scatter(x=df['Sales'], y=df['Profit'], mode='markers', 
              name='Sales vs Profit', marker=dict(color='coral', size=5)),
    row=1, col=2
)

# Sales by Product Type
product_sales = df.groupby('Product_Type')['Sales'].sum()
fig.add_trace(
    go.Bar(x=product_sales.index, y=product_sales.values, 
           name='Product Sales', marker_color='lightgreen'),
    row=2, col=1
)

# Sales over time
df_time = df.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')
fig.add_trace(
    go.Scatter(x=df_time['Date'], y=df_time['Sales'], mode='lines+markers',
              name='Time Series', marker_color='purple'),
    row=2, col=2
)

fig.update_layout(
    height=800,
    title_text="Interactive Sales Analysis Dashboard",
    template='plotly_white',
    showlegend=False
)

fig.update_xaxes(title_text="Sales ($)", row=1, col=1)
fig.update_xaxes(title_text="Sales ($)", row=1, col=2)
fig.update_xaxes(title_text="Product Type", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=2)

fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Profit ($)", row=1, col=2)
fig.update_yaxes(title_text="Total Sales ($)", row=2, col=1)
fig.update_yaxes(title_text="Total Sales ($)", row=2, col=2)

fig.show()
```

### 5.2 Animated Visualizations: Time in Motion

```python
# Plotly animated scatter plot
fig = px.scatter(df, x='Sales', y='Profit',
                 animation_frame=df['Date'].dt.strftime('%Y-%m'),
                 animation_group='Product_Type',
                 color='Product_Type',
                 size='Marketing',
                 hover_name='Product_Type',
                 title='Animated: Sales vs Profit Over Time',
                 labels={'Sales': 'Sales ($)', 'Profit': 'Profit ($)'},
                 range_x=[df['Sales'].min() - 10, df['Sales'].max() + 10],
                 range_y=[df['Profit'].min() - 10, df['Profit'].max() + 10])

fig.update_layout(template='plotly_white')
fig.show()

# Animated bar chart
df_monthly = df.copy()
df_monthly['Year_Month'] = df_monthly['Date'].dt.to_period('M').astype(str)
monthly_sales = df_monthly.groupby(['Year_Month', 'Product_Type'])['Sales'].sum().reset_index()

fig = px.bar(monthly_sales, x='Product_Type', y='Sales',
             animation_frame='Year_Month',
             color='Product_Type',
             title='Animated: Monthly Sales by Product Type',
             labels={'Sales': 'Total Sales ($)', 'Product_Type': 'Product Type'},
             range_y=[0, monthly_sales['Sales'].max() * 1.1])

fig.update_layout(template='plotly_white', showlegend=False)
fig.show()
```

---

## Part 6: Practical Case Study
### Complete Sales Analysis Dashboard

Let's put everything together in a comprehensive analysis:

```python
# Create a comprehensive analysis dashboard
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Sales Distribution by Product Type', 
                    'Sales vs Profit Relationship',
                    'Monthly Sales Trend',
                    'Profit Margin by Product Type',
                    'Marketing Efficiency',
                    'Correlation Heatmap'),
    specs=[[{"type": "box"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "heatmap"}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. Sales distribution by product type
for i, product in enumerate(df['Product_Type'].unique()):
    product_data = df[df['Product_Type'] == product]['Sales']
    fig.add_trace(
        go.Box(y=product_data, name=product, showlegend=False),
        row=1, col=1
    )

# 2. Sales vs Profit scatter
fig.add_trace(
    go.Scatter(x=df['Sales'], y=df['Profit'], mode='markers',
              marker=dict(color=df['Marketing'], size=8, 
                         colorscale='Viridis', showscale=True,
                         colorbar=dict(title="Marketing", x=1.15)),
              name='Sales vs Profit', showlegend=False,
              hovertemplate='Sales: $%{x:.2f}<br>Profit: $%{y:.2f}<br>Marketing: $%{marker.color:.2f}<extra></extra>'),
    row=1, col=2
)

# 3. Monthly sales trend
df['Year_Month'] = df['Date'].dt.to_period('M').astype(str)
monthly_trend = df.groupby('Year_Month')['Sales'].sum().reset_index()
fig.add_trace(
    go.Scatter(x=monthly_trend['Year_Month'], y=monthly_trend['Sales'],
              mode='lines+markers', name='Monthly Sales',
              line=dict(width=3, color='steelblue'),
              marker=dict(size=8)),
    row=2, col=1
)

# 4. Profit margin by product type
df['Profit_Margin_Pct'] = (df['Profit'] / df['Sales']) * 100
profit_margin = df.groupby('Product_Type')['Profit_Margin_Pct'].mean().sort_values(ascending=False)
fig.add_trace(
    go.Bar(x=profit_margin.index, y=profit_margin.values,
           marker_color='coral', name='Avg Profit Margin', showlegend=False),
    row=2, col=2
)

# 5. Marketing efficiency (Sales per Marketing dollar)
df['Marketing_Efficiency'] = df['Sales'] / df['Marketing']
efficiency = df.groupby('Product_Type')['Marketing_Efficiency'].mean().sort_values(ascending=False)
fig.add_trace(
    go.Bar(x=efficiency.index, y=efficiency.values,
           marker_color='lightgreen', name='Marketing Efficiency', showlegend=False),
    row=3, col=1
)

# 6. Correlation heatmap
numeric_cols = ['Sales', 'Profit', 'COGS', 'Marketing']
corr_matrix = df[numeric_cols].corr()
fig.add_trace(
    go.Heatmap(z=corr_matrix.values,
               x=corr_matrix.columns,
               y=corr_matrix.columns,
               colorscale='RdBu',
               zmid=0,
               text=corr_matrix.values,
               texttemplate='%{text:.2f}',
               textfont={"size":10},
               showscale=True,
               colorbar=dict(title="Correlation", x=1.15)),
    row=3, col=2
)

# Update layout
fig.update_layout(
    height=1200,
    title_text="<b>Comprehensive Sales Analysis Dashboard</b>",
    title_font_size=20,
    template='plotly_white',
    showlegend=False
)

# Update axes labels
fig.update_xaxes(title_text="Product Type", row=1, col=1)
fig.update_xaxes(title_text="Sales ($)", row=1, col=2)
fig.update_xaxes(title_text="Month", row=2, col=1, tickangle=45)
fig.update_xaxes(title_text="Product Type", row=2, col=2, tickangle=45)
fig.update_xaxes(title_text="Product Type", row=3, col=1, tickangle=45)
fig.update_xaxes(title_text="Variable", row=3, col=2)

fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
fig.update_yaxes(title_text="Profit ($)", row=1, col=2)
fig.update_yaxes(title_text="Total Sales ($)", row=2, col=1)
fig.update_yaxes(title_text="Profit Margin (%)", row=2, col=2)
fig.update_yaxes(title_text="Sales per Marketing $", row=3, col=1)
fig.update_yaxes(title_text="Variable", row=3, col=2)

fig.show()
```

---

## Part 7: Best Practices and Tips

### 7.1 Choosing the Right Visualization

- **Univariate continuous**: Histogram, box plot, violin plot
- **Univariate categorical**: Bar chart, pie chart (sparingly)
- **Bivariate continuous**: Scatter plot, line chart
- **Bivariate categorical**: Grouped bar chart, stacked bar chart
- **Multivariate**: Scatter matrix, 3D scatter, faceted plots, bubble charts

### 7.2 Color Considerations

```python
# Good color palettes
# Sequential (for ordered data)
colors_seq = px.colors.sequential.Viridis

# Diverging (for data with center point)
colors_div = px.colors.diverging.RdBu

# Qualitative (for categorical data)
colors_qual = px.colors.qualitative.Set3

# Colorblind-friendly palettes
sns.color_palette("colorblind")
```

### 7.3 Saving Visualizations

```python
# Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(df['Sales'], df['Profit'], 'o')
plt.savefig('sales_profit.png', dpi=300, bbox_inches='tight')
plt.close()

# Seaborn
sns.scatterplot(data=df, x='Sales', y='Profit')
plt.savefig('sales_profit_seaborn.png', dpi=300, bbox_inches='tight')
plt.close()

# Plotly
fig = px.scatter(df, x='Sales', y='Profit')
fig.write_html('sales_profit.html')  # Interactive HTML
fig.write_image('sales_profit.png', width=1200, height=600, scale=2)  # Static image
```

### 7.4 Performance Tips

- For large datasets, sample data before plotting
- Use `alpha` parameter to handle overlapping points
- Consider aggregation for time series with many points
- Use Plotly's `visible=False` for complex dashboards

---

## Key Takeaways

1. **Start Simple**: Begin with univariate visualizations to understand your data
2. **Build Complexity**: Progress to bivariate, then multivariate as you understand relationships
3. **Choose Wisely**: Match visualization type to your data and question
4. **Tell a Story**: Every visualization should answer a question or reveal an insight
5. **Interactivity Matters**: Plotly's interactivity helps explore data deeply
6. **Beauty with Purpose**: Aesthetics enhance understanding, but clarity is paramount
7. **Iterate**: Your first visualization is rarely your last—refine based on insights

---

## Next Steps

After mastering data visualization:

1. **Advanced Plotly**: Explore dashboards with Dash framework
2. **Geographic Visualizations**: Maps and spatial data visualization
3. **Statistical Visualizations**: Confidence intervals, regression plots
4. **Custom Styling**: Create publication-ready visualizations
5. **Interactive Dashboards**: Build web applications for data exploration
6. **Machine Learning Visualizations**: Feature importance, model performance

---

## Resources for Continued Learning

- **Matplotlib Gallery**: matplotlib.org/gallery
- **Seaborn Tutorial**: seaborn.pydata.org/tutorial.html
- **Plotly Documentation**: plotly.com/python
- **Data Visualization Best Practices**: "The Visual Display of Quantitative Information" by Edward Tufte
- **Color Theory**: colorbrewer2.org for color palette selection

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*
