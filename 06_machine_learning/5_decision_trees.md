# Decision Trees and Random Forest

## What is a Decision Tree?

A decision tree is like asking a series of **yes/no questions** to make a decision.

**Think of it like this:**
- You want to decide if you should go outside
- Question 1: Is it raining? 
  - If YES → Stay inside
  - If NO → Ask next question
- Question 2: Is it sunny?
  - If YES → Go outside
  - If NO → Maybe stay inside

**In Machine Learning:**
- Each question = **Internal Node**
- Each answer path = **Branch** 
- Final decision = **Leaf Node**

## Decision Tree Structure

```
Root Node (Income > $50k?)
├── YES: Go to next question (Age > 30?)
│   ├── YES: Buy House (LEAF)
│   └── NO: Don't Buy (LEAF)
└── NO: Don't Buy House (LEAF)
```

**Components:**
- **Root Node**: Starting point with best splitting feature
- **Internal Nodes**: Questions that split the data
- **Branches**: Possible answers (YES/NO for binary splits)
- **Leaf Nodes**: Final predictions

**Tree Structure Visualization:**
```
                Root Node
              (Income > $50k?)
                   |
          ┌────────┴────────┐
          │                 │
        YES                NO
          │                 │
    Internal Node      Leaf Node
    (Age > 30?)       (Don't Buy)
          |
     ┌────┴────┐
     │         │
   YES        NO
     │         │
 Leaf Node  Leaf Node
(Buy House) (Don't Buy)
```

## How Does a Decision Tree Learn?

**Step 1:** Start with all data at root

**Step 2:** Find the **best question** to split data
- Which feature separates classes best?
- Use metrics like Gini Impurity or Entropy

**Step 3:** Split data based on that question

**Step 4:** Repeat for each subset until:
- All data points have same label
- No more useful splits possible
- Maximum depth reached

## Simple Example - Will You Play Tennis?

**Dataset:**
| Weather | Temperature | Play Tennis |
|---------|-------------|-------------|
| Sunny   | Hot         | No          |
| Sunny   | Cool        | Yes         |
| Rainy   | Cool        | Yes         |
| Rainy   | Hot         | No          |

**Decision Tree:**
```
Weather?
├── Sunny → Temperature?
│   ├── Hot → No
│   └── Cool → Yes  
└── Rainy → Temperature?
    ├── Hot → No
    └── Cool → Yes
```

## Splitting Criteria - How to Choose Best Split?

**Gini Impurity**: Measures how "impure" a split is
- 0 = Pure (all same class)
- 0.5 = Maximum impurity (50-50 split)

**Entropy**: Measures information gain
- 0 = Pure 
- 1 = Maximum uncertainty

**Goal**: Choose split that **reduces impurity the most**

## Decision Tree Code Example

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample data
data = {
    'Income': [50000, 30000, 70000, 45000, 80000],
    'Age': [25, 35, 45, 28, 50],
    'Children': [0, 2, 1, 0, 3],
    'Buy_House': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[['Income', 'Age', 'Children']]
y = df['Buy_House']

# Create and train decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# Make predictions
predictions = dt.predict(X)
print("Predictions:", predictions)
```

## Visualizing Decision Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt, 
          feature_names=['Income', 'Age', 'Children'],
          class_names=['No', 'Yes'], 
          filled=True, 
          rounded=True)
plt.title("Decision Tree: Will Buy House?")
plt.show()
```

**What you'll see:**
- Boxes showing the splitting condition
- Colors indicating the majority class
- Numbers showing samples in each node

## The Problem - Overfitting!

**Decision trees can become too complex:**
- They memorize training data
- Poor performance on new data
- Very deep trees with many branches

**Example of Overfitting:**
```
Income > $50k?
├── YES → Age > 30?
│   ├── YES → Children > 2?
│   │   ├── YES → Day = Monday? (TOO SPECIFIC!)
│   │   └── NO → Buy House
│   └── NO → Don't Buy
└── NO → Don't Buy
```

## Preventing Overfitting

**Techniques:**

**Max Depth**: Limit tree height
- `max_depth=5`

**Min Samples Split**: Minimum samples needed to split
- `min_samples_split=10`

**Min Samples Leaf**: Minimum samples in leaf node
- `min_samples_leaf=5`

```python
# Better decision tree with constraints
dt_improved = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```


## Complete Example - Building Decision Tree with Gini

Let's work through a complete example using the "Cool As Ice" dataset:

**Original Dataset:**
| Loves Popcorn | Loves Soda | Age | Cool As Ice |
|---------------|------------|-----|-------------|
| Yes           | Yes        | 7   | No          |
| Yes           | No         | 12  | No          |
| No            | Yes        | 18  | Yes         |
| No            | Yes        | 35  | Yes         |
| Yes           | Yes        | 38  | Yes         |
| Yes           | No         | 50  | No          |
| No            | No         | 83  | No          |

**Target Distribution:**
- Cool As Ice = Yes: 3 samples
- Cool As Ice = No: 4 samples
- Total: 7 samples

## Step 1: Calculate Root Gini Impurity

**Root Gini Impurity:**
```
Gini = 1 - (P_yes)² - (P_no)²
Gini = 1 - (3/7)² - (4/7)²
Gini = 1 - (0.429)² - (0.571)²
Gini = 1 - 0.184 - 0.327
Gini = 0.489
```

## Step 2: Calculate Gini for Each Feature Split

**Feature 1: Loves Popcorn**
- **Loves Popcorn = Yes**: [No, No, Yes, No] → 1 Yes, 3 No
  - Gini = 1 - (1/4)² - (3/4)² = 1 - 0.063 - 0.563 = 0.375
- **Loves Popcorn = No**: [Yes, Yes, No] → 2 Yes, 1 No  
  - Gini = 1 - (2/3)² - (1/3)² = 1 - 0.444 - 0.111 = 0.444

**Weighted Gini for Loves Popcorn:**
```
Weighted_Gini = (4/7) × 0.375 + (3/7) × 0.444 = 0.405
```

**Feature 2: Loves Soda**
- **Loves Soda = Yes**: [No, Yes, Yes, Yes] → 3 Yes, 1 No
  - Gini = 1 - (3/4)² - (1/4)² = 1 - 0.563 - 0.063 = 0.375
- **Loves Soda = No**: [No, No, No] → 0 Yes, 3 No
  - Gini = 1 - (0/3)² - (3/3)² = 1 - 0 - 1 = 0.0

**Weighted Gini for Loves Soda:**
```
Weighted_Gini = (4/7) × 0.375 + (3/7) × 0.0 = 0.214
```

**Feature 3: Age (let's try Age < 15)**
- **Age < 15**: [No, No] → 0 Yes, 2 No
  - Gini = 1 - (0/2)² - (2/2)² = 0.0
- **Age ≥ 15**: [Yes, Yes, Yes, No, No] → 3 Yes, 2 No
  - Gini = 1 - (3/5)² - (2/5)² = 1 - 0.36 - 0.16 = 0.48

**Weighted Gini for Age < 15:**
```
Weighted_Gini = (2/7) × 0.0 + (5/7) × 0.48 = 0.343
```

## Step 3: Choose Best Split

**Information Gain = Root Gini - Weighted Gini**

- **Loves Popcorn**: 0.489 - 0.405 = 0.084
- **Loves Soda**: 0.489 - 0.214 = 0.275 ← **HIGHEST**
- **Age < 15**: 0.489 - 0.343 = 0.146

**Winner**: Loves Soda (lowest Gini = 0.214, highest information gain = 0.275)

## Step 4: Build the Complete Tree

```
Root: Loves Soda?
├── TRUE (Loves Soda = Yes): [7, 12, 18, 35, 38] → [No, No, Yes, Yes, Yes]
│   │   Need further split: Age < 12.5?
│   ├── TRUE (Age < 12.5): [7, 12] → [No, No] 
│   │   └── LEAF: Cool As Ice = No
│   └── FALSE (Age ≥ 12.5): [18, 35, 38] → [Yes, Yes, Yes]
│       └── LEAF: Cool As Ice = Yes
│
└── FALSE (Loves Soda = No): [12, 50, 83] → [No, No, No]
    └── LEAF: Cool As Ice = No
```

## Step 5: Code Implementation

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create the dataset
data = {
    'Loves_Popcorn': [1, 1, 0, 0, 1, 1, 0],
    'Loves_Soda': [1, 0, 1, 1, 1, 0, 0], 
    'Age': [7, 12, 18, 35, 38, 50, 83],
    'Cool_As_Ice': [0, 0, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
X = df[['Loves_Popcorn', 'Loves_Soda', 'Age']]
y = df['Cool_As_Ice']

print("Dataset:")
print(df)

# Create decision tree with Gini criterion
dt = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=3,
    min_samples_leaf=1,
    random_state=42
)

dt.fit(X, y)

# Plot the tree
plt.figure(figsize=(15, 10))
plot_tree(dt, 
          feature_names=['Loves_Popcorn', 'Loves_Soda', 'Age'],
          class_names=['Not Cool', 'Cool As Ice'], 
          filled=True, 
          rounded=True,
          fontsize=12)
plt.title("Decision Tree: Cool As Ice Prediction")
plt.show()

# Make predictions
predictions = dt.predict(X)
print("\nPredictions vs Actual:")
comparison = pd.DataFrame({
    'Actual': y,
    'Predicted': predictions,
    'Correct': y == predictions
})
print(comparison)
```

## Step 6: Understanding the Results

**Final Tree Interpretation:**
1. **First Question**: Do you love soda?
   - If **NO** → You're **NOT Cool As Ice**
   - If **YES** → Go to next question

2. **Second Question**: Are you younger than 12.5?
   - If **YES** → You're **NOT Cool As Ice** 
   - If **NO** → You're **Cool As Ice**

**Key Insights:**
- Loving soda is the most important factor
- Among soda lovers, age matters (older = cooler!)
- The tree perfectly classifies this small dataset
- But with only 1 sample in some leaves, it might overfit

## Why This Tree Might Overfit

**Problem with Small Leaf Nodes:**
- Some leaves have very few samples (like Age < 12.5 with only 1 sample)
- Hard to have confidence in predictions for new data
- Tree memorized specific cases rather than learning patterns

**Solutions:**
- Use `min_samples_leaf=2` or higher
- Collect more data
- Use ensemble methods like Random Forest

