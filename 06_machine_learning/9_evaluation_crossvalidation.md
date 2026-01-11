# Machine Learning Model Evaluation and Optimization
## Complete Lecture with Practical Examples

---

## Table of Contents
1. [Model Evaluation for Classification](#classification-evaluation)
2. [Model Evaluation for Regression](#regression-evaluation)
3. [Overfitting vs Underfitting](#overfitting-underfitting)
4. [Bias vs Variance](#bias-variance)
5. [Cross Validation](#cross-validation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)

---

## 1. Model Evaluation for Classification {#classification-evaluation}

### What is Classification Evaluation?

Classification evaluation measures how well your model predicts categories or classes.

Think of it like grading a student's multiple choice test - you want to know not just the final score, but which types of questions they got right or wrong.

### Key Classification Metrics

#### Accuracy
- **Definition**: Percentage of correct predictions out of total predictions
- **Formula**: (Correct Predictions) / (Total Predictions)
- **Example**: If your spam filter correctly identifies 95 out of 100 emails, accuracy = 95%

#### Precision
- **Definition**: Out of all positive predictions, how many were actually correct?
- **Formula**: True Positives / (True Positives + False Positives)
- **Example**: Out of 100 emails your filter marked as spam, 90 were actually spam → Precision = 90%

#### Recall (Sensitivity)
- **Definition**: Out of all actual positive cases, how many did we catch?
- **Formula**: True Positives / (True Positives + False Negatives)
- **Example**: Out of 80 actual spam emails, your filter caught 72 → Recall = 90%

#### F1-Score
- **Definition**: Harmonic mean of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Use Case**: When you need balance between precision and recall

#### Confusion Matrix
- **Definition**: Table showing correct vs incorrect predictions for each class
- **Components**:
  - True Positives (TP): Correctly predicted positive
  - True Negatives (TN): Correctly predicted negative
  - False Positives (FP): Incorrectly predicted positive
  - False Negatives (FN): Incorrectly predicted negative

### Practical Example: Email Spam Detection

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Sample data (in practice, use real email features)
# Features: [email_length, num_links, num_caps, num_exclamation]
X = np.random.rand(1000, 4)
y = np.random.randint(0, 2, 1000)  # 0: Not Spam, 1: Spam

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### ROC Curve and AUC
- **ROC Curve**: Plot of True Positive Rate vs False Positive Rate
- **AUC**: Area Under the ROC Curve (higher is better)
- **Interpretation**: AUC = 0.5 means random guessing, AUC = 1.0 means perfect classifier

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get prediction probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print(f"AUC Score: {roc_auc:.3f}")
```

---

## 2. Model Evaluation for Regression {#regression-evaluation}

### What is Regression Evaluation?

Regression evaluation measures how close your predicted numbers are to the actual numbers.

Think of it like measuring how close your guess of house prices is to the actual selling prices.

### Key Regression Metrics

#### Mean Absolute Error (MAE)
- **Definition**: Average of absolute differences between predicted and actual values
- **Formula**: Σ|actual - predicted| / n
- **Example**: If house prices differ by $10K, $20K, $15K on average, MAE = $15K

#### Mean Squared Error (MSE)
- **Definition**: Average of squared differences between predicted and actual values
- **Formula**: Σ(actual - predicted)² / n
- **Advantage**: Penalizes large errors more heavily

#### Root Mean Squared Error (RMSE)
- **Definition**: Square root of MSE
- **Formula**: √(MSE)
- **Advantage**: Same units as the target variable

#### R² Score (Coefficient of Determination)
- **Definition**: Proportion of variance in target variable explained by the model
- **Range**: -∞ to 1 (1 is perfect, 0 means no better than mean)
- **Interpretation**: R² = 0.85 means model explains 85% of the variance

### Practical Example: House Price Prediction

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Sample data (in practice, use real house features)
# Features: [size_sqft, bedrooms, age, location_score]
X = np.random.rand(1000, 4) * 100  # Scale for realistic values
# Target: house prices (in thousands)
y = X[:, 0] * 2 + X[:, 1] * 10 + np.random.normal(0, 10, 1000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:.2f}K")
print(f"Mean Squared Error: ${mse:.2f}K²")
print(f"Root Mean Squared Error: ${rmse:.2f}K")
print(f"R² Score: {r2:.3f}")

# Interpretation
print(f"\nInterpretation:")
print(f"On average, predictions are off by ${mae:.2f}K")
print(f"Model explains {r2*100:.1f}% of price variance")
```

### Residual Analysis

```python
import matplotlib.pyplot as plt

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(12, 4))

# Residuals vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

# Histogram of residuals
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.tight_layout()
plt.show()
```

---

## 3. Overfitting vs Underfitting {#overfitting-underfitting}

### Understanding the Concepts

Think of learning to drive:
- **Underfitting**: Like a driving student who only knows "press gas pedal" - too simple, poor performance everywhere
- **Overfitting**: Like memorizing every pothole on your practice route but failing on new roads - perfect on training, poor on new data
- **Good Fit**: Understanding traffic rules and adapting to new situations

### Underfitting (High Bias)

#### Characteristics:
- **Training Performance**: Poor
- **Test Performance**: Poor (similar to training)
- **Model Complexity**: Too simple
- **Learning**: Model hasn't captured the underlying pattern

#### Causes:
- Model is too simple for the data
- Not enough features
- Too much regularization
- Insufficient training time

#### Real-world Example:
Using linear regression to predict stock prices - stocks don't follow straight lines!

### Overfitting (High Variance)

#### Characteristics:
- **Training Performance**: Excellent
- **Test Performance**: Poor (much worse than training)
- **Model Complexity**: Too complex
- **Learning**: Model memorized training data instead of learning patterns

#### Causes:
- Model is too complex for available data
- Too many features
- Not enough training data
- Too little regularization

#### Real-world Example:
A medical diagnosis system that memorizes every patient in training data but fails on new patients.

### Practical Example: Polynomial Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 1, 50).reshape(-1, 1)
y = 2 * X.ravel() + 0.5 * X.ravel()**2 + np.random.normal(0, 0.1, 50)

# Split data
X_train, X_test = X[:35], X[35:]
y_train, y_test = y[:35], y[35:]

# Test different polynomial degrees
degrees = [1, 2, 9]
colors = ['blue', 'green', 'red']
labels = ['Underfit (degree=1)', 'Good Fit (degree=2)', 'Overfit (degree=9)']

plt.figure(figsize=(15, 5))

for i, (degree, color, label) in enumerate(zip(degrees, colors, labels)):
    # Create polynomial model
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Train model
    poly_model.fit(X_train, y_train)
    
    # Generate smooth curve for plotting
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    y_plot = poly_model.predict(X_plot)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, poly_model.predict(X_train))
    test_error = mean_squared_error(y_test, poly_model.predict(X_test))
    
    # Plot
    plt.subplot(1, 3, i+1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
    plt.scatter(X_test, y_test, color='orange', alpha=0.6, label='Test Data')
    plt.plot(X_plot, y_plot, color=color, linewidth=2, label=f'{label}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'{label}\nTrain MSE: {train_error:.3f}, Test MSE: {test_error:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Detecting Overfitting and Underfitting

```python
from sklearn.model_selection import learning_curve

# Generate learning curves
def plot_learning_curves(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    # Convert to positive MSE
    train_scores = -train_scores
    val_scores = -val_scores
    
    # Calculate means and stds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Error')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Error')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Learning Curves: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example with different models
models = [
    (Pipeline([('poly', PolynomialFeatures(degree=1)), ('linear', LinearRegression())]), 'Underfitting'),
    (Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]), 'Good Fit'),
    (Pipeline([('poly', PolynomialFeatures(degree=9)), ('linear', LinearRegression())]), 'Overfitting')
]

for model, title in models:
    plot_learning_curves(model, X, y, title)
```

### Solutions

#### For Underfitting:
- Use more complex models
- Add more features
- Reduce regularization
- Train for more epochs (neural networks)

#### For Overfitting:
- Get more training data
- Use simpler models
- Add regularization
- Use cross-validation
- Feature selection
- Early stopping (neural networks)

---

## 4. Bias vs Variance {#bias-variance}

### Understanding Bias and Variance

Think of learning to predict something as trying to hit a target. The target represents the true relationship in your data.

**Imagine you're playing darts:**
- **High Bias**: Your darts consistently hit the same wrong spot (systematic error)
- **High Variance**: Your darts scatter all around the target (inconsistent)
- **Low Bias, Low Variance**: Your darts cluster around the bullseye (ideal)

### Bias (Underfitting)

#### Definition:
Systematic error that occurs when your model makes overly simplistic assumptions.

#### Real-World Example 1: Weather Prediction
- **High Bias Model**: "Temperature tomorrow = Temperature today"
- **Problem**: Always predicts same pattern, misses seasonal changes, weather fronts
- **Result**: Consistently wrong in predictable ways (always underestimates winter warming trends)

#### Real-World Example 2: House Price Prediction
- **High Bias Model**: "Price = $100 per square foot" (ignoring location, age, condition)
- **Problem**: Treats Manhattan apartment same as rural farmhouse
- **Result**: Always underestimates expensive areas, overestimates cheap areas

#### Real-World Example 3: Medical Diagnosis
- **High Bias Model**: "If fever > 100°F → Flu" (ignoring other symptoms)
- **Problem**: Misses diseases that also cause fever (infections, COVID, etc.)
- **Result**: Consistently misdiagnoses non-flu conditions with fever

#### Characteristics of High Bias:
- **Systematic Error**: Model consistently makes the same type of mistakes
- **Simple Models**: Linear models trying to fit curved relationships
- **Consistent Performance**: Similar (poor) performance on different datasets
- **Cannot Capture Complexity**: Misses important patterns in data
- **Underfitting**: Model is too simple for the problem

### Variance (Overfitting)

#### Definition:
Random error that occurs when your model is too sensitive to small changes in training data.

#### Real-World Example 1: Stock Market Prediction
- **High Variance Model**: Complex model memorizing every daily price movement
- **Problem**: Creates specific rule for each historical event
- **Result**: Predicts differently based on which days were in training set

#### Real-World Example 2: Image Recognition
- **High Variance Model**: Deep neural network trained on few images
- **Problem**: Learns specific details of training images (lighting, angle, background)
- **Result**: Fails when new images have slightly different conditions

#### Real-World Example 3: Customer Behavior Prediction
- **High Variance Model**: Decision tree with 50 levels of depth
- **Problem**: Creates unique rule for each customer in training data
- **Result**: Performs very differently when trained on different customer samples

#### Characteristics of High Variance:
- **Random Error**: Model performance varies significantly between training sets
- **Complex Models**: High-degree polynomials, very deep decision trees
- **Inconsistent Performance**: Very different results on different training sets
- **Over-sensitive**: Small changes in training data cause big changes in model
- **Overfitting**: Model memorizes training data instead of learning patterns

### The Four Combinations Explained

#### 1. High Bias, Low Variance
**Example**: Linear regression predicting house prices using only square footage

**What happens:**
- Every time you train the model, it learns roughly the same straight line
- The line consistently misses the curved relationship between size and price
- Performance is predictable but consistently poor

**Like a doctor who:**
- Always diagnoses "stress" for any symptom
- Is consistent but systematically wrong
- Misses serious conditions

#### 2. Low Bias, High Variance  
**Example**: Very deep decision tree with no pruning

**What happens:**
- Each training session produces a completely different tree structure
- Sometimes performs brilliantly, sometimes terribly
- Learns training data perfectly but fails on new data

**Like a doctor who:**
- Studies each patient's case history in extreme detail
- Makes different diagnoses for similar symptoms based on irrelevant details
- Sometimes brilliant, often completely wrong

#### 3. High Bias, High Variance
**Example**: Poorly designed neural network with wrong architecture

**What happens:**
- Model is too simple to capture patterns (high bias)
- Also unstable across different training sets (high variance)
- Worst of both worlds

**Like a doctor who:**
- Uses overly simplified diagnostic rules
- But applies them inconsistently
- Both systematically and randomly wrong

#### 4. Low Bias, Low Variance (The Goal)
**Example**: Well-tuned Random Forest with proper parameters

**What happens:**
- Captures the true underlying relationship
- Performs consistently across different training sets
- Generalizes well to new data

**Like a doctor who:**
- Uses comprehensive, evidence-based diagnostic criteria
- Makes consistent, accurate diagnoses
- Adapts appropriately to new cases

### Practical Examples Across Different Domains

#### Marketing Campaign Prediction

**High Bias Example:**
- Model: "Email campaigns always have 2% response rate"
- Problem: Ignores email content, timing, audience, seasonality
- Result: Always predicts 2%, whether sending cat videos or major sales announcements

**High Variance Example:**
- Model: Complex neural network trained on 100 past campaigns
- Problem: Learns that "Tuesday email about red products to customers named John" has 15% response rate
- Result: Makes wildly different predictions for slight campaign changes

**Balanced Example:**
- Model: Considers email subject, customer segments, timing, past engagement
- Result: Reliable predictions that adapt to campaign variations

#### Sports Performance Prediction

**High Bias Example:**
- Model: "Team with higher salary always wins"
- Problem: Ignores current form, injuries, matchups, weather
- Result: Always favors expensive teams, misses upsets

**High Variance Example:**
- Model: Learns every detail of past games (referee, grass length, fans' clothing colors)
- Problem: Makes predictions based on irrelevant patterns
- Result: Wildly different predictions for similar matchups

**Balanced Example:**
- Model: Uses player statistics, recent form, head-to-head records, injuries
- Result: Consistent, reasonable predictions that account for key factors

### The Mathematical Relationship

**Total Error = Bias² + Variance + Irreducible Error**

#### Where:
- **Bias²**: How far off your average prediction is from the true value
- **Variance**: How much your predictions vary from their average
- **Irreducible Error**: Natural noise in the data that can't be eliminated

#### The Tradeoff:
- **Reducing Bias**: Usually increases variance (more complex models)
- **Reducing Variance**: Usually increases bias (simpler, more constrained models)
- **Goal**: Find the sweet spot that minimizes total error

### Real-World Impact Examples

#### Healthcare
- **High Bias**: Simple checklist misses complex conditions → Patients suffer
- **High Variance**: Over-complex diagnostic tool gives inconsistent results → Doctor loses trust
- **Balanced**: Reliable diagnostic aid that catches most conditions consistently

#### Finance
- **High Bias**: Simple credit scoring misses good candidates → Lost business opportunities
- **High Variance**: Complex model gives different scores for same person → Regulatory problems
- **Balanced**: Fair, consistent credit decisions

#### Autonomous Vehicles
- **High Bias**: Simple "stop if obstacle" rule → Can't handle complex traffic
- **High Variance**: Over-complex system reacts differently to similar situations → Unpredictable, dangerous
- **Balanced**: Consistent, appropriate responses to traffic situations

### How to Identify Bias vs Variance in Your Models

#### Signs of High Bias:
- Training and validation errors are both high
- Training and validation errors are close to each other
- Model performs similarly poorly on different datasets
- Learning curves plateau at high error levels

#### Signs of High Variance:
- Large gap between training and validation error
- Training error is low, validation error is high
- Model performance varies dramatically between training runs
- Small changes in training data cause big changes in predictions

#### Signs of Good Balance:
- Training and validation errors are both reasonably low
- Small gap between training and validation error
- Consistent performance across different training sets
- Learning curves converge to reasonable error levels

### Managing Bias-Variance Tradeoff

#### Reducing Bias:
- Use more complex models
- Add more features
- Reduce regularization parameters
- Increase model capacity

#### Reducing Variance:
- Get more training data
- Use ensemble methods
- Add regularization
- Use cross-validation
- Feature selection

#### Practical Approach:

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import validation_curve

# Example: Finding optimal max_depth for Random Forest
param_range = range(1, 15)
train_scores, val_scores = validation_curve(
    RandomForestRegressor(n_estimators=100, random_state=42), 
    X, y, param_name='max_depth', param_range=param_range,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Convert to positive MSE
train_scores = -train_scores
val_scores = -val_scores

# Calculate means
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Error')
plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Error')
plt.xlabel('Max Depth')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff: Random Forest Max Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Find optimal depth
optimal_depth = param_range[np.argmin(val_mean)]
print(f"Optimal max_depth: {optimal_depth}")
```

---

## 5. Cross Validation {#cross-validation}

### What is Cross Validation?

Cross validation is like getting multiple opinions before making a decision.

Instead of testing your model on just one test set, you test it on multiple different splits of your data to get a more reliable estimate of performance.

Think of it as taking several practice exams instead of just one to better predict your actual exam performance.

### Why Use Cross Validation?

#### Problems with Single Train-Test Split:
- **Lucky/Unlucky Splits**: Performance might depend on which data ended up in test set
- **Unreliable Estimates**: One test set might not represent true performance
- **Data Waste**: Large portion of data sits unused in test set

#### Benefits of Cross Validation:
- **More Reliable**: Multiple evaluations give better performance estimate
- **Uses All Data**: Every sample gets to be in both training and test sets
- **Reduces Variance**: Average performance across folds is more stable
- **Better Model Selection**: More confident in choosing between models

### Types of Cross Validation

#### 1. K-Fold Cross Validation

**How it works:**
1. Split data into K equal parts (folds)
2. Train on K-1 folds, test on remaining fold
3. Repeat K times, each fold serves as test set once
4. Average the K performance scores

**Common choices:**
- K=5: Good balance of bias and variance
- K=10: Lower bias, but higher computational cost
- K=n (Leave-One-Out): Lowest bias, highest computational cost

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                         n_redundant=10, n_clusters_per_class=1, random_state=42)

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("5-Fold Cross Validation Results:")
print(f"Individual fold scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.3f}")
print(f"Standard deviation: {cv_scores.std():.3f}")
print(f"95% confidence interval: {cv_scores.mean():.3f} ± {1.96 * cv_scores.std():.3f}")

# Detailed cross-validation with KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

print("\nDetailed Fold Results:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Train and evaluate
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_val_fold, y_val_fold)
    fold_results.append(score)
    
    print(f"Fold {fold + 1}: {score:.3f} (Train size: {len(train_idx)}, Val size: {len(val_idx)})")

print(f"\nOverall CV Score: {np.mean(fold_results):.3f} ± {np.std(fold_results):.3f}")
```

#### 2. Stratified K-Fold Cross Validation

**When to use:**
Imbalanced datasets where you want to maintain the same proportion of samples from each class in every fold.

**Example:**
If your dataset has 80% class A and 20% class B, each fold will also have roughly 80% class A and 20% class B.

```python
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                         weights=[0.8, 0.2], random_state=42)

print("Original class distribution:")
print(pd.Series(y).value_counts(normalize=True))

# Regular K-Fold
regular_kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\nRegular K-Fold class distributions:")
for fold, (train_idx, val_idx) in enumerate(regular_kf.split(X)):
    val_distribution = pd.Series(y[val_idx]).value_counts(normalize=True)
    print(f"Fold {fold + 1}: {dict(val_distribution)}")

# Stratified K-Fold
stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nStratified K-Fold class distributions:")
for fold, (train_idx, val_idx) in enumerate(stratified_kf.split(X, y)):
    val_distribution = pd.Series(y[val_idx]).value_counts(normalize=True)
    print(f"Fold {fold + 1}: {dict(val_distribution)}")

# Compare performance
regular_scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
stratified_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

print(f"\nRegular K-Fold: {regular_scores.mean():.3f} ± {regular_scores.std():.3f}")
print(f"Stratified K-Fold: {stratified_scores.mean():.3f} ± {stratified_scores.std():.3f}")
```

#### 3. Leave-One-Out Cross Validation (LOOCV)

**How it works:**
- Use n-1 samples for training, 1 sample for testing
- Repeat n times (where n is total number of samples)
- Average the n performance scores

**Pros and Cons:**
- **Pros**: Uses maximum data for training, lowest bias
- **Cons**: Very expensive computationally, high variance

```python
from sklearn.model_selection import LeaveOneOut

# Use smaller dataset for demonstration (LOOCV is expensive)
X_small, y_small = X[:100], y[:100]

# Leave-One-Out Cross Validation
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X_small, y_small, cv=loo)

print(f"Leave-One-Out CV Results:")
print(f"Number of folds: {len(loo_scores)}")
print(f"Mean accuracy: {loo_scores.mean():.3f}")
print(f"Standard deviation: {loo_scores.std():.3f}")

# Compare with 5-fold CV
five_fold_scores = cross_val_score(model, X_small, y_small, cv=5)
print(f"\n5-Fold CV for comparison:")
print(f"Mean accuracy: {five_fold_scores.mean():.3f}")
print(f"Standard deviation: {five_fold_scores.std():.3f}")
```

#### 4. Time Series Cross Validation

**When to use:**
When working with time series data where the order matters.

**How it works:**
- Use past data to predict future data
- Maintain temporal order (no data leakage from future)

```python
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Generate time series data
n_samples = 200
time_index = np.arange(n_samples)
trend = 0.02 * time_index
seasonal = np.sin(2 * np.pi * time_index / 20)
noise = np.random.normal(0, 0.1, n_samples)
y_ts = trend + seasonal + noise
X_ts = time_index.reshape(-1, 1)

# Time Series Cross Validation
tscv = TimeSeriesSplit(n_splits=5)

print("Time Series Cross Validation Splits:")
plt.figure(figsize=(12, 8))

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    print(f"Fold {fold + 1}: Train {train_idx[0]}-{train_idx[-1]}, Test {test_idx[0]}-{test_idx[-1]}")
    
    plt.subplot(3, 2, fold + 1)
    plt.plot(time_index[train_idx], y_ts[train_idx], 'blue', label='Train', alpha=0.7)
    plt.plot(time_index[test_idx], y_ts[test_idx], 'red', label='Test', alpha=0.7)
    plt.title(f'Fold {fold + 1}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluate time series model
from sklearn.linear_model import LinearRegression
ts_scores = cross_val_score(LinearRegression(), X_ts, y_ts, cv=tscv)
print(f"\nTime Series CV Score: {ts_scores.mean():.3f} ± {ts_scores.std():.3f}")
```

### Cross Validation for Model Selection

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Compare multiple models using cross validation
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

cv_results = {}

print("Model Comparison using 5-Fold Cross Validation:")
print("="*60)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_results[name] = scores
    
    print(f"{name:20}: {scores.mean():.3f} ± {scores.std():.3f}")

# Visualize results
plt.figure(figsize=(10, 6))
box_data = [scores for scores in cv_results.values()]
plt.boxplot(box_data, labels=list(cv_results.keys()))
plt.title('Model Performance Comparison (5-Fold CV)')
plt.ylabel('Accuracy Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Statistical significance test
from scipy import stats

best_model = max(cv_results.items(), key=lambda x: x[1].mean())
print(f"\nBest Model: {best_model[0]} ({best_model[1].mean():.3f} ± {best_model[1].std():.3f})")

# Compare best model with others
for name, scores in cv_results.items():
    if name != best_model[0]:
        t_stat, p_value = stats.ttest_rel(best_model[1], scores)
        significance = "significantly better" if p_value < 0.05 else "not significantly different"
        print(f"{best_model[0]} vs {name}: p-value = {p_value:.4f} ({significance})")
```

### Cross Validation Best Practices

#### 1. Choosing K
- **K=5 or K=10**: Most common choices, good balance
- **Small datasets**: Use larger K (more data for training)
- **Large datasets**: Use smaller K (computational efficiency)
- **Very small datasets**: Consider Leave-One-Out

#### 2. Multiple Metrics
```python
from sklearn.model_selection import cross_validate

# Evaluate multiple metrics simultaneously
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

print("Multiple Metrics Cross Validation:")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize():12}: {scores.mean():.3f} ± {scores.std():.3f}")
```

#### 3. Nested Cross Validation
```python
from sklearn.model_selection import GridSearchCV

# Nested CV for unbiased model selection and evaluation
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None]
}

# Inner loop: hyperparameter tuning
clf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                   param_grid=param_grid, cv=inner_cv, scoring='accuracy')

# Outer loop: performance estimation
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='accuracy')

print("Nested Cross Validation Results:")
print(f"CV Score: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
print("This gives an unbiased estimate of model performance")
```

---

## 6. Hyperparameter Tuning {#hyperparameter-tuning}

### What are Hyperparameters?

**Hyperparameters** are configuration settings for machine learning algorithms that you set before training begins.

Think of them as the "settings" on your camera - ISO, aperture, shutter speed. You adjust these settings to get the best photo, just like you adjust hyperparameters to get the best model performance.

**Examples:**
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`
- **SVM**: `C`, `gamma`, `kernel`
- **Neural Networks**: `learning_rate`, `batch_size`, `number_of_layers`

### Parameters vs Hyperparameters

#### Parameters:
- **Definition**: Learned by the algorithm from data
- **Examples**: Weights in neural networks, coefficients in linear regression
- **Control**: Algorithm finds optimal values automatically

#### Hyperparameters:
- **Definition**: Set by the practitioner before training
- **Examples**: Learning rate, number of trees, regularization strength
- **Control**: You must choose these values

### Why Tune Hyperparameters?

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                         n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare default vs tuned hyperparameters
print("Impact of Hyperparameter Tuning:")
print("="*40)

# Default hyperparameters
default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X_train, y_train)
default_score = accuracy_score(y_test, default_rf.predict(X_test))
print(f"Default hyperparameters: {default_score:.3f}")

# Better hyperparameters (found through tuning)
tuned_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
tuned_rf.fit(X_train, y_train)
tuned_score = accuracy_score(y_test, tuned_rf.predict(X_test))
print(f"Tuned hyperparameters:   {tuned_score:.3f}")
print(f"Improvement:             {tuned_score - default_score:.3f}")
```

### Types of Hyperparameter Tuning

#### 1. Grid Search

**How it works:**
- Define a grid of hyperparameter values
- Try every combination
- Choose the combination with best cross-validation score

**Pros:**
- Guaranteed to find the best combination in the grid
- Easy to understand and implement
- Systematic and thorough

**Cons:**
- Computationally expensive (exponential growth)
- Curse of dimensionality
- May miss optimal values between grid points

```python
from sklearn.model_selection import GridSearchCV
import time

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print(f"Grid Search will test {np.prod([len(v) for v in param_grid.values()])} combinations")

# Perform grid search
start_time = time.time()
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5-fold cross validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1   # Show progress
)

grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nGrid Search completed in {end_time - start_time:.2f} seconds")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Test performance on test set
best_model = grid_search.best_estimator_
test_score = accuracy_score(y_test, best_model.predict(X_test))
print(f"Test set performance: {test_score:.3f}")

# Analyze results
results_df = pd.DataFrame(grid_search.cv_results_)
print(f"\nTop 5 parameter combinations:")
top_5 = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
for idx, row in top_5.iterrows():
    print(f"Score: {row['mean_test_score']:.3f} ± {row['std_test_score']:.3f}, Params: {row['params']}")
```

#### 2. Random Search

**How it works:**
- Define distributions for each hyperparameter
- Randomly sample combinations
- Try a fixed number of random combinations

**Pros:**
- More efficient than grid search
- Can find good solutions quickly
- Works well when only a few hyperparameters matter

**Cons:**
- No guarantee of finding the best combination
- Results can vary between runs
- May miss optimal regions

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),           # Random integers between 50-300
    'max_depth': [3, 5, 7, 10, None],          # Discrete choices
    'min_samples_split': randint(2, 20),       # Random integers between 2-20
    'min_samples_leaf': randint(1, 10),        # Random integers between 1-10
    'max_features': ['sqrt', 'log2', None],    # Discrete choices
    'bootstrap': [True, False]                 # Boolean choices
}

# Perform random search
start_time = time.time()
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,  # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nRandom Search completed in {end_time - start_time:.2f} seconds")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.3f}")

# Compare with grid search
print(f"\nComparison:")
print(f"Grid Search:   {grid_search.best_score_:.3f} in {end_time - start_time:.1f}s")
print(f"Random Search: {random_search.best_score_:.3f} in {end_time - start_time:.1f}s")
```

#### 3. Bayesian Optimization (Advanced)

**How it works:**
- Build a probabilistic model of the objective function
- Use this model to decide where to search next
- Update model with each new result

**Pros:**
- Very efficient for expensive evaluations
- Intelligent search strategy
- Can handle continuous and discrete parameters

**Cons:**
- More complex to set up
- Requires additional libraries
- May get stuck in local optima

**When to use:**
- When model training is very expensive (deep learning, large datasets)
- When you have limited computational budget
- When you want to find good parameters quickly

**Popular libraries:**
- scikit-optimize
- Optuna
- Hyperopt

### Hyperparameter Tuning for Different Algorithms

#### Random Forest Hyperparameters

```python
# Random Forest hyperparameters with explanations
rf_params = {
    'n_estimators': [50, 100, 200, 300],        # Number of trees
    'max_depth': [3, 5, 7, 10, None],           # Maximum depth of trees
    'min_samples_split': [2, 5, 10],            # Min samples to split node
    'min_samples_leaf': [1, 2, 4],              # Min samples in leaf node
    'max_features': ['sqrt', 'log2', None],     # Features per split
    'bootstrap': [True, False]                  # Bootstrap sampling
}

print("Random Forest Hyperparameter Guide:")
print("="*40)
print("n_estimators: More trees = better performance but slower")
print("max_depth: Controls overfitting (None = no limit)")
print("min_samples_split: Higher = less overfitting")
print("min_samples_leaf: Higher = smoother decision boundary")
print("max_features: 'sqrt' often works well for classification")
print("bootstrap: True for bagging, False for deterministic")
```

#### Support Vector Machine (SVM) Hyperparameters

```python
from sklearn.svm import SVC

# SVM hyperparameters
svm_params = {
    'C': [0.1, 1, 10, 100, 1000],              # Regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], # Kernel coefficient
    'kernel': ['rbf', 'poly', 'sigmoid']        # Kernel type
}

# Tune SVM
svm_grid = GridSearchCV(
    SVC(random_state=42),
    svm_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("\nSVM Hyperparameter Guide:")
print("="*30)
print("C: Regularization strength (higher C = less regularization)")
print("gamma: Kernel coefficient (higher = more complex boundary)")
print("kernel: rbf (default), poly (polynomial), sigmoid")
print(f"\nBest SVM params: {svm_grid.best_params_}")
print(f"Best SVM score: {svm_grid.best_score_:.3f}")
```

#### Gradient Boosting Hyperparameters

```python
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting hyperparameters
gb_params = {
    'n_estimators': [50, 100, 200],             # Number of boosting stages
    'learning_rate': [0.01, 0.1, 0.2, 0.3],    # Shrinks contribution
    'max_depth': [3, 4, 5, 6],                  # Depth of individual trees
    'min_samples_split': [2, 5, 10],            # Min samples to split
    'min_samples_leaf': [1, 2, 4],              # Min samples in leaf
    'subsample': [0.8, 0.9, 1.0]               # Fraction of samples per tree
}

# Use RandomizedSearchCV for efficiency
gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

gb_random.fit(X_train, y_train)

print("\nGradient Boosting Hyperparameter Guide:")
print("="*42)
print("n_estimators: Number of boosting rounds")
print("learning_rate: Lower = more conservative learning")
print("max_depth: Depth of each tree (3-6 often good)")
print("subsample: Fraction of data per tree (0.8-1.0)")
print(f"\nBest GB params: {gb_random.best_params_}")
print(f"Best GB score: {gb_random.best_score_:.3f}")
```

### Hyperparameter Tuning Best Practices

#### 1. Start Simple, Then Optimize
```python
# Example: Progressive hyperparameter tuning

# Step 1: Coarse grid search
coarse_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

coarse_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                            coarse_params, cv=3, n_jobs=-1)
coarse_search.fit(X_train, y_train)

print("Step 1 - Coarse search:")
print(f"Best params: {coarse_search.best_params_}")

# Step 2: Fine-tune around best parameters
best_n_est = coarse_search.best_params_['n_estimators']
best_depth = coarse_search.best_params_['max_depth']

fine_params = {
    'n_estimators': [best_n_est - 50, best_n_est, best_n_est + 50],
    'max_depth': [best_depth - 2, best_depth, best_depth + 2] if best_depth else [8, 10, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

fine_search = GridSearchCV(RandomForestClassifier(random_state=42),
                          fine_params, cv=5, n_jobs=-1)
fine_search.fit(X_train, y_train)

print("\nStep 2 - Fine-tuned search:")
print(f"Best params: {fine_search.best_params_}")
print(f"Improvement: {fine_search.best_score_ - coarse_search.best_score_:.4f}")
```

#### 2. Use Appropriate Cross-Validation
```python
# Stratified CV for imbalanced datasets
# Time series CV for temporal data
# Group CV for grouped data

from sklearn.model_selection import StratifiedKFold, GroupKFold

# Example with stratified CV
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stratified_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
    cv=stratified_cv,  # Use stratified CV
    scoring='f1_weighted'  # Appropriate metric for imbalanced data
)

stratified_grid.fit(X_train, y_train)
print(f"Stratified CV result: {stratified_grid.best_score_:.3f}")
```

#### 3. Monitor for Overfitting
```python
# Validation curves to understand hyperparameter effects
from sklearn.model_selection import validation_curve

# Analyze effect of n_estimators
param_range = [50, 100, 150, 200, 250, 300]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), X, y,
    param_name='n_estimators', param_range=param_range,
    cv=5, scoring='accuracy', n_jobs=-1
)

# Plot validation curve
plt.figure(figsize=(10, 6))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

plt.xlabel('n_estimators')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve: n_estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Find optimal point
optimal_idx = np.argmax(val_mean)
optimal_n_estimators = param_range[optimal_idx]
print(f"Optimal n_estimators: {optimal_n_estimators}")
print(f"Training score: {train_mean[optimal_idx]:.3f}")
print(f"Validation score: {val_mean[optimal_idx]:.3f}")
```

#### 4. Resource Management
```python
# Example of resource-conscious hyperparameter tuning

# For large datasets or expensive models
quick_params = {
    'n_estimators': [50, 100],      # Fewer options
    'max_depth': [5, 10]            # Limited depth
}

# Use smaller CV folds for speed
quick_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    quick_params,
    cv=3,           # Fewer folds
    n_jobs=-1,      # Use all cores
    verbose=0       # Less output
)

# For production: more thorough search
production_params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

production_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    production_params,
    n_iter=100,     # More iterations
    cv=5,           # More folds
    n_jobs=-1,
    random_state=42
)

print("Resource Management:")
print("- Quick search for development/prototyping")
print("- Thorough search for production models")
print("- Use RandomizedSearchCV for large parameter spaces")
print("- Monitor computational time and resources")
```

### Summary: Complete Workflow

```python
# Complete hyperparameter tuning workflow
def tune_model(X, y, test_size=0.2):
    """Complete hyperparameter tuning workflow"""
    
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 2. Define parameter space
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 3. Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4. Perform hyperparameter search
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # 5. Fit and get best model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # 6. Evaluate on test set
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    # 7. Return results
    return {
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'train_score': train_score,
        'test_score': test_score,
        'overfitting': train_score - test_score
    }

# Example usage
results = tune_model(X, y)

print("Final Model Performance:")
print("="*30)
print(f"Best parameters: {results['best_params']}")
print(f"CV score: {results['cv_score']:.3f}")
print(f"Train score: {results['train_score']:.3f}")
print(f"Test score: {results['test_score']:.3f}")
print(f"Overfitting gap: {results['overfitting']:.3f}")

if results['overfitting'] > 0.05:
    print("Warning: Model might be overfitting!")
else:
    print("Model shows good generalization.")
```

---

## Key Takeaways

### Model Evaluation
- **Classification**: Use accuracy, precision, recall, F1-score, and AUC-ROC based on your specific problem
- **Regression**: Use MAE for interpretability, RMSE for penalizing large errors, R² for explained variance
- **Always** use appropriate metrics for your problem domain and business requirements

### Overfitting vs Underfitting
- **Underfitting**: Model too simple, add complexity or features
- **Overfitting**: Model too complex, add regularization or more data
- **Use learning curves** to diagnose and validation curves to find optimal complexity

### Bias vs Variance
- **High Bias**: Systematic errors, use more complex models
- **High Variance**: Inconsistent predictions, use regularization or ensemble methods
- **Tradeoff**: Find the sweet spot through cross-validation and proper model selection

### Cross Validation
- **Always use CV** for reliable performance estimates
- **Choose appropriate CV**: K-fold for general use, stratified for imbalanced data, time series for temporal data
- **Nested CV** for unbiased model selection and evaluation

### Hyperparameter Tuning
- **Start simple**, then optimize systematically
- **Use appropriate search strategy**: Grid search for small spaces, random search for large spaces
- **Monitor resources** and avoid overfitting to validation set
- **Always evaluate** final model on held-out test set

---

**Remember**: Machine learning is an iterative process. Start with simple baselines, understand your data, and gradually improve through systematic evaluation and optimization!