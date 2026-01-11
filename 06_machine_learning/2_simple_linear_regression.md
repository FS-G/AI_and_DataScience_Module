# Regression Basics

## Introduction to Regression

Regression is a fundamental statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (predictors). It helps us understand how changes in input variables affect the output and allows us to make predictions.

### Key Concepts

**Dependent Variable (Y)**: The outcome we want to predict or explain

**Independent Variable (X)**: The predictor variable used to explain or predict Y

**Regression Line**: The best-fit line through the data points

**Residuals**: The differences between observed and predicted values

## Common Dataset for All Examples

Throughout this guide, we'll use a consistent dataset of **Advertising Spend vs Sales Revenue** to demonstrate all concepts. This dataset contains 20 observations of monthly advertising spending (in thousands of dollars) and corresponding sales revenue (in thousands of dollars) for a small business.

### Dataset: Advertising Spend vs Sales Revenue

| Month | Advertising_Spend | Sales_Revenue |
|-------|------------------|---------------|
| 1     | 2.5              | 15.2          |
| 2     | 3.1              | 18.7          |
| 3     | 1.8              | 12.5          |
| 4     | 4.2              | 24.1          |
| 5     | 2.9              | 17.3          |
| 6     | 3.8              | 21.9          |
| 7     | 2.1              | 14.8          |
| 8     | 4.5              | 26.3          |
| 9     | 3.3              | 19.6          |
| 10    | 2.7              | 16.1          |
| 11    | 4.1              | 23.7          |
| 12    | 1.9              | 13.2          |
| 13    | 3.6              | 20.8          |
| 14    | 2.3              | 15.6          |
| 15    | 4.7              | 27.1          |
| 16    | 3.0              | 18.2          |
| 17    | 2.6              | 16.7          |
| 18    | 4.3              | 25.4          |
| 19    | 1.7              | 11.9          |
| 20    | 3.9              | 22.5          |

**Variables:**
- **X (Independent)**: Advertising_Spend (thousands of dollars)
- **Y (Dependent)**: Sales_Revenue (thousands of dollars)

This dataset will allow us to answer: "How does advertising spending affect sales revenue?"

## Simple Linear Regression

Simple linear regression examines the relationship between two continuous variables, where one variable is used to predict another.

### The Mathematical Foundation

The relationship between variables X and Y can be expressed as:

```
Y = β₀ + β₁X + ε
```

Where:
- **Y**: Dependent variable (Sales Revenue)
- **X**: Independent variable (Advertising Spend)
- **β₀**: Y-intercept (expected sales with $0 advertising)
- **β₁**: Slope (change in sales for $1000 increase in advertising)
- **ε**: Error term (random variation not explained by the model)

### Parameter Estimation Using Least Squares Method

The goal is to find the values of β₀ and β₁ that minimize the sum of squared residuals.

#### Slope (β₁) Calculation:
```
β₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
```

Alternative formula:
```
β₁ = [n·Σ(xᵢyᵢ) - Σ(xᵢ)·Σ(yᵢ)] / [n·Σ(xᵢ²) - (Σ(xᵢ))²]
```

#### Intercept (β₀) Calculation:
```
β₀ = ȳ - β₁·x̄
```

Where:
- x̄ = mean of Advertising Spend values
- ȳ = mean of Sales Revenue values
- n = number of observations (20)

### Understanding the Parameters

**β₀ (Intercept)**: The expected sales revenue when advertising spend equals zero. This represents the baseline sales level without advertising.

**β₁ (Slope)**: The expected change in sales revenue for a one-unit ($1000) increase in advertising spend. This quantifies the return on advertising investment:
- Positive β₁: More advertising leads to higher sales
- Negative β₁: More advertising leads to lower sales (unlikely in this context)
- β₁ = 0: No relationship between advertising and sales

## Model Evaluation Metrics

### R-squared (Coefficient of Determination)
```
R² = 1 - (SSres / SStot)
```

Where:
- SSres = Σ(yᵢ - ŷᵢ)² (Sum of squared residuals)
- SStot = Σ(yᵢ - ȳ)² (Total sum of squares)

R² represents the proportion of variance in sales revenue explained by advertising spend (ranges from 0 to 1).

### Mean Squared Error (MSE)
```
MSE = Σ(yᵢ - ŷᵢ)² / n
```

### Root Mean Squared Error (RMSE)
```
RMSE = √(MSE)
```

## Implementation in Python with Scikit-learn

### Complete Implementation with Our Dataset

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create the advertising dataset
data = {
    'Month': range(1, 21),
    'Advertising_Spend': [2.5, 3.1, 1.8, 4.2, 2.9, 3.8, 2.1, 4.5, 3.3, 2.7, 
                         4.1, 1.9, 3.6, 2.3, 4.7, 3.0, 2.6, 4.3, 1.7, 3.9],
    'Sales_Revenue': [15.2, 18.7, 12.5, 24.1, 17.3, 21.9, 14.8, 26.3, 19.6, 16.1, 
                     23.7, 13.2, 20.8, 15.6, 27.1, 18.2, 16.7, 25.4, 11.9, 22.5]
}

df = pd.DataFrame(data)
print("Advertising Spend vs Sales Revenue Dataset:")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")

# Prepare features and target
X = df[['Advertising_Spend']]  # Features (must be 2D for sklearn)
y = df['Sales_Revenue']        # Target

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Extract parameters
intercept = model.intercept_
slope = model.coef_[0]

print(f"\n=== MODEL PARAMETERS ===")
print(f"Intercept (β₀): {intercept:.4f}")
print(f"Slope (β₁): {slope:.4f}")

# Interpretation
print(f"\n=== INTERPRETATION ===")
print(f"Expected sales with $0 advertising: ${intercept:.2f}k")
print(f"Expected sales increase per $1k advertising: ${slope:.2f}k")

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n=== MODEL EVALUATION ===")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Make predictions for new values
new_ad_spends = np.array([[2.0], [3.5], [5.0]])
predictions = model.predict(new_ad_spends)
print(f"\n=== PREDICTIONS ===")
for spend, pred in zip(new_ad_spends.flatten(), predictions):
    print(f"Advertising spend: ${spend:.1f}k → Predicted sales: ${pred:.2f}k")
```

### Visualization

```python
# Create comprehensive visualization
plt.figure(figsize=(15, 10))

# Plot 1: All data with regression line
plt.subplot(2, 2, 1)
X_range = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)

plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Training data', s=60)
plt.scatter(X_test, y_test, color='green', alpha=0.7, label='Test data', s=60)
plt.plot(X_range, y_range_pred, color='red', linewidth=2, label=f'Regression line (R²={test_r2:.3f})')
plt.xlabel('Advertising Spend ($k)')
plt.ylabel('Sales Revenue ($k)')
plt.title('Advertising Spend vs Sales Revenue')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals vs Fitted
plt.subplot(2, 2, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, color='purple', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values (Predicted Sales)')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.grid(True, alpha=0.3)

# Plot 3: Histogram of residuals
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=8, color='orange', alpha=0.7, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)

# Plot 4: Actual vs Predicted
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_test_pred, color='darkblue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales Revenue ($k)')
plt.ylabel('Predicted Sales Revenue ($k)')
plt.title('Actual vs Predicted Values')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Implementation with Statsmodels

Statsmodels provides more detailed statistical output, including p-values and confidence intervals.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Using our advertising dataset
print("=== STATSMODELS REGRESSION ANALYSIS ===")

# Method 1: Using formula API (more intuitive)
model_formula = smf.ols('Sales_Revenue ~ Advertising_Spend', data=df).fit()

# Method 2: Using arrays (alternative approach)
X_with_intercept = sm.add_constant(X)  # Adds intercept term
model_array = sm.OLS(y, X_with_intercept).fit()

# Display comprehensive results
print("\nDETAILED STATISTICAL RESULTS:")
print("="*50)
print(model_formula.summary())

# Extract key statistics
print(f"\n=== KEY STATISTICS ===")
print(f"R-squared: {model_formula.rsquared:.4f}")
print(f"Adjusted R-squared: {model_formula.rsquared_adj:.4f}")
print(f"F-statistic: {model_formula.fvalue:.4f}")
print(f"F-statistic p-value: {model_formula.f_pvalue:.6f}")

# Extract coefficients and their statistics
params = model_formula.params
pvalues = model_formula.pvalues
conf_int = model_formula.conf_int()

print(f"\n=== COEFFICIENT ANALYSIS ===")
print(f"Intercept: {params['Intercept']:.4f} (p={pvalues['Intercept']:.4f})")
print(f"Advertising Spend: {params['Advertising_Spend']:.4f} (p={pvalues['Advertising_Spend']:.4f})")

print(f"\n=== 95% CONFIDENCE INTERVALS ===")
print("Intercept: [{:.4f}, {:.4f}]".format(conf_int.iloc[0, 0], conf_int.iloc[0, 1]))
print("Advertising Spend: [{:.4f}, {:.4f}]".format(conf_int.iloc[1, 0], conf_int.iloc[1, 1]))

# Predictions with confidence intervals
predictions = model_formula.get_prediction(X)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% confidence

print(f"\n=== PREDICTIONS WITH CONFIDENCE INTERVALS (First 5 rows) ===")
pred_df = pd.DataFrame({
    'Advertising_Spend': X.flatten(),
    'Actual_Sales': y,
    'Predicted_Sales': pred_summary['mean'],
    'CI_Lower': pred_summary['mean_ci_lower'],
    'CI_Upper': pred_summary['mean_ci_upper']
})
print(pred_df.head().round(3))
```

### Understanding the Statsmodels Output

The summary() output provides crucial statistical information:

**Model Statistics**:
- **R-squared**: Proportion of variance explained (higher is better)
- **Adj. R-squared**: R-squared adjusted for number of parameters
- **F-statistic**: Tests overall model significance
- **Prob (F-statistic)**: P-value for overall model

**Coefficients Table**:
- **coef**: Parameter estimates (β₀, β₁)
- **std err**: Standard error of the coefficient
- **t**: t-statistic for hypothesis testing
- **P>|t|**: P-value for the coefficient
- **[0.025, 0.975]**: 95% confidence interval

### P-Value Interpretation for Our Dataset

**P-value**: The probability of observing the current result (or more extreme) if the null hypothesis were true.

**For Individual Coefficients**:
- H₀: βᵢ = 0 (no relationship between advertising and sales)
- H₁: βᵢ ≠ 0 (significant relationship exists)

**Interpretation Guidelines**:
- p < 0.001: Very strong evidence against H₀ (highly significant)
- p < 0.01: Strong evidence against H₀ (very significant) 
- p < 0.05: Moderate evidence against H₀ (significant)
- p < 0.10: Weak evidence against H₀ (marginally significant)
- p ≥ 0.10: Insufficient evidence to reject H₀ (not significant)

**For our advertising example**: If p < 0.05 for advertising spend, we can conclude that advertising spending has a statistically significant effect on sales revenue.

### Confidence Intervals

A 95% confidence interval means we are 95% confident that the true parameter value lies within this range.

For our advertising example: If the 95% CI for the slope is [3.2, 5.8], we're 95% confident that each $1000 increase in advertising spending increases sales revenue by between $3200 and $5800.


## Practical Example: Complete Analysis

### Problem Setup
We want to understand: **"How does advertising spending affect sales revenue for our business?"**

**Business Context**:
- Small retail business tracking monthly performance
- Advertising spend ranges from $1.7k to $4.7k per month
- Sales revenue ranges from $11.9k to $27.1k per month
- Goal: Optimize advertising budget for maximum ROI

### Expected Results Analysis

Based on our advertising dataset, here's what we typically find:

**Model Parameters** (approximate):
- β₀ ≈ 3.5 (p < 0.05)
- β₁ ≈ 5.2 (p < 0.001) 
- R² ≈ 0.85

**Business Interpretation**:

1. **Intercept (β₀ = 3.5)**:
   - With $0 advertising spend, expected sales are $3,500
   - This represents baseline sales from existing customers, word-of-mouth, etc.
   - Statistically significant (p < 0.05) indicating meaningful baseline sales

2. **Slope (β₁ = 5.2)**:
   - Each $1,000 increase in advertising generates $5,200 additional sales revenue
   - ROI = 520% (every dollar spent on advertising generates $5.20 in sales)
   - Highly significant (p < 0.001) indicating strong advertising effectiveness

3. **Model Fit (R² = 0.85)**:
   - 85% of sales variation is explained by advertising spending
   - Strong predictive power for business planning
   - Remaining 15% due to other factors (seasonality, competition, etc.)

**Business Decisions**:
- **Increase advertising budget**: High ROI suggests more advertising is profitable
- **Budget planning**: Can reliably predict sales based on advertising spend
- **Performance monitoring**: Significant relationship allows tracking advertising effectiveness

### Confidence Intervals and Practical Significance

If 95% CI for slope is [4.1, 6.3]:
- We're 95% confident true ROI is between 410% and 630%
- Even the lower bound (410%) represents excellent advertising effectiveness
- Narrow confidence interval indicates precise estimate

**Business Risk Assessment**:
- **Best case** (slope = 6.3): $6,300 return per $1,000 advertising
- **Worst case** (slope = 4.1): $4,100 return per $1,000 advertising  
- **Both scenarios** are highly profitable, reducing investment risk

## Linear Regression Assumptions

Understanding these assumptions is crucial for building reliable models and interpreting results correctly.

### 1. Linearity
**What it means**: The relationship between advertising spend and sales revenue is linear.

**Why it matters**: If the true relationship is curved (e.g., diminishing returns at high advertising levels), a straight line won't fit well.

**How to check**: Scatter plot should show roughly straight-line pattern. For our advertising data, look for consistent rate of sales increase across all advertising levels.

### 2. Independence of Observations  
**What it means**: Each month's data should be independent. One month's performance shouldn't directly influence another.

**Why it matters**: If months are related (e.g., advertising effects carry over), standard errors will be incorrect.

**Common violations in business data**: 
- Seasonal patterns
- Carryover advertising effects
- Economic trends affecting multiple months

### 3. Homoscedasticity (Constant Variance)
**What it means**: The spread of residuals should be roughly the same whether advertising spend is low ($2k) or high ($5k).

**Why it matters**: If variance changes, predictions will be more reliable for some advertising levels than others.

**Business example**: If prediction errors are larger for high advertising spends, we're less certain about ROI at higher budget levels.

### 4. Normality of Residuals
**What it means**: The prediction errors should be approximately normally distributed.

**Why it matters**: Needed for hypothesis tests and confidence intervals to be valid.

**How to check**: Histogram of residuals should look bell-shaped. Q-Q plot should show points roughly following straight line.

### 5. No Extreme Outliers
**What it means**: No months with extremely unusual advertising spend or sales combinations.

**Business context**: Outliers might be:
- Month with major promotion affecting sales differently
- Data entry errors
- Unusual external events (competitor closure, supply issues)

### 6. No Perfect Multicollinearity (For Multiple Regression)
**What it means**: Independent variables shouldn't be perfectly correlated with each other.

**Note**: Not applicable to simple regression, but important when adding variables like seasonality, competitor spending, etc.

### Diagnostic Plots for Our Dataset

```python
# Enhanced diagnostic plots for advertising data
def plot_diagnostics(X, y, model, title_prefix=""):
    """Create comprehensive diagnostic plots"""
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Residuals vs Fitted
    plt.subplot(2, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.7, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Fitted Values (Predicted Sales)')
    plt.ylabel('Residuals')
    plt.title(f'{title_prefix}Residuals vs Fitted Values')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plot
    from scipy import stats
    plt.subplot(2, 3, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{title_prefix}Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Scale-Location plot
    plt.subplot(2, 3, 3)
    plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.7, color='green')
    plt.xlabel('Fitted Values')
    plt.ylabel('√|Residuals|')
    plt.title(f'{title_prefix}Scale-Location Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of residuals
    plt.subplot(2, 3, 4)
    plt.hist(residuals, bins=8, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix}Histogram of Residuals')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Leverage plot (for identifying influential points)
    plt.subplot(2, 3, 5)
    n = len(X)
    p = X.shape[1] + 1  # number of parameters including intercept
    leverage = []
    
    # Calculate leverage for each point
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    H = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    leverages = np.diag(H)
    
    plt.scatter(range(len(leverages)), leverages, alpha=0.7, color='purple')
    plt.axhline(y=2*p/n, color='red', linestyle='--', label=f'Threshold: {2*p/n:.3f}')
    plt.xlabel('Observation Index')
    plt.ylabel('Leverage')
    plt.title(f'{title_prefix}Leverage Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Cook's Distance
    plt.subplot(2, 3, 6)
    # Calculate Cook's distance
    mse = np.mean(residuals**2)
    cooks_d = []
    for i in range(len(X)):
        X_i = np.delete(X_with_intercept, i, 0)
        y_i = np.delete(y, i, 0)
        try:
            beta_i = np.linalg.inv(X_i.T @ X_i) @ X_i.T @ y_i
            y_pred_i = X_with_intercept[i] @ beta_i
            cooks_d.append((residuals[i]**2 / (p * mse)) * (leverages[i] / (1 - leverages[i])**2))
        except:
            cooks_d.append(0)
    
    plt.scatter(range(len(cooks_d)), cooks_d, alpha=0.7, color='red')
    plt.axhline(y=4/n, color='red', linestyle='--', label=f'Threshold: {4/n:.3f}')
    plt.xlabel('Observation Index')
    plt.ylabel("Cook's Distance")
    plt.title(f"{title_prefix}Cook's Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print diagnostic summary
    print(f"\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Mean of residuals: {np.mean(residuals):.6f} (should be ~0)")
    print(f"Std deviation of residuals: {np.std(residuals):.4f}")
    print(f"Shapiro-Wilk test p-value: {stats.shapiro(residuals)[1]:.4f} (>0.05 for normality)")
    print(f"Number of high leverage points: {sum(leverages > 2*p/n)} (threshold: {2*p/n:.3f})")
    print(f"Number of influential points: {sum(np.array(cooks_d) > 4/n)} (Cook's D > {4/n:.3f})")

# Apply to our advertising data
plot_diagnostics(X, y, model, "Advertising Data - ")
```
