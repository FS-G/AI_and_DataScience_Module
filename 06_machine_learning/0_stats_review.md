# Statistics Review - Basic Concepts

## 1. Types of Data

### Categorical Data
- Data that represents categories or groups
- Examples: Gender (Male/Female), Color (Red/Blue/Green), Yes/No responses
- Cannot perform mathematical operations (cannot add or subtract categories)
- Can be:
  - **Nominal**: Categories with no order (e.g., colors, cities)
  - **Ordinal**: Categories with order (e.g., ratings: Poor/Fair/Good/Excellent)

### Continuous Data
- Data that can take any value within a range
- Examples: Height, Weight, Temperature, Age, Income
- Can perform mathematical operations (can add, subtract, multiply, divide)
- Also called numerical or quantitative data
- Can be:
  - **Interval**: Differences are meaningful, but no true zero (e.g., temperature in Celsius)
  - **Ratio**: Has a true zero point (e.g., height, weight, income)

## 2. Central Tendency

Central tendency measures describe the "center" or typical value of a dataset.

### Mean
- The average of all values
- Formula: Sum of all values ÷ Number of values
- Example: [10, 20, 30, 40, 50] → Mean = (10+20+30+40+50)/5 = 30
- Sensitive to outliers (extreme values)

### Median
- The middle value when data is sorted in order
- If odd number of values: middle value
- If even number of values: average of two middle values
- Example: [10, 20, 30, 40, 50] → Median = 30
- Less sensitive to outliers than mean

### Mode
- The value that appears most frequently
- Can have one mode, multiple modes, or no mode
- Example: [10, 20, 20, 30, 40] → Mode = 20
- Useful for categorical data

## 3. P-value

### What is a P-value?
- A probability value that helps determine if results are statistically significant
- Range: 0 to 1
- Lower p-value = stronger evidence against the null hypothesis

### Common Interpretation
- **p < 0.05**: Results are statistically significant (less than 5% chance results occurred by random chance)
- **p ≥ 0.05**: Results are not statistically significant (results could be due to random chance)

### Example
- If p-value = 0.03, there's only a 3% chance the observed results happened by random chance
- If p-value = 0.20, there's a 20% chance the observed results happened by random chance

### Important Notes
- P-value does NOT tell you:
  - The size of the effect
  - Whether the result is practically important
  - The probability that your hypothesis is true
- P-value only tells you: How likely your results are if there's actually no effect (null hypothesis is true)
