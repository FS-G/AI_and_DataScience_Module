# Logistic Regression: Complete Lecture Guide

## 1. Regression vs Classification - Quick Overview

### **Regression**
- **Purpose**: Predicts continuous numerical values
- **Output**: Real numbers (e.g., house prices, temperature)
- **Example**: Predicting stock prices based on market indicators

### **Classification** 
- **Purpose**: Predicts discrete categories or classes
- **Output**: Class labels (e.g., spam/not spam, malignant/benign)
- **Example**: Email classification, medical diagnosis

---

## 2. Classification Problems in Real World

### **Common Applications**
- **Email Spam Detection**: Spam vs Not Spam
- **Medical Diagnosis**: Disease vs Healthy
- **Image Recognition**: Cat vs Dog
- **Credit Approval**: Approve vs Reject
- **Sentiment Analysis**: Positive vs Negative

### **Popular Classification Algorithms**
- **Logistic Regression** (Our Focus)
- **Decision Trees**
- **Random Forest**
- **Support Vector Machines (SVM)**
- **Naive Bayes**
- **Neural Networks**

---

## 3. The Logistic Function (Sigmoid Function)

### **Definition**
- Maps any **real-valued number** to a value between **0 and 1**
- Useful for **outputting probabilities** (always between 0 and 1)

### **Mathematical Form**
```
σ(z) = 1 / (1 + e^(-z))
```

**Where:**
- `z` = input to the function (any real number)
- `e` = Euler's number (≈ 2.718)
- `σ(z)` = output between 0 and 1

### **Properties**
- **S-shaped curve**
- **Outputs probability** (between 0 and 1)

### **Real-Life Analogy**
Think of sigmoid like a **smart switch** that converts input voltages (real numbers) into light brightness between 0 (off) and 1 (fully on). The probability tells us how likely the light is to be on based on voltage.

### **Behavior**
- **As z approaches ∞**: σ(z) → 1
- **As z approaches -∞**: σ(z) → 0

```python
# Plotting Sigmoid Function
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate data
x = np.arange(-10, 10, 0.1)
y = sigmoid(x)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.grid(True)
plt.show()
```

---

## 4. Logistic Regression Model

### **Model Equation**
```
p(y = 1|x) = σ(z) = σ(w·x + b)
```

**Where:**
- `w` = weights vector
- `x` = input features vector  
- `b` = bias term

### **Interpretation**
- **z = w·x + b**: Linear combination of inputs and weights
- **σ(z)**: Probability that y = 1

### **Practical Example: Email Spam Detection**

```
z = w₁ × (Number of "free") + w₂ × (Subject length) + b
```

Then apply sigmoid:
```
p(spam|features) = 1 / (1 + e^(-z))
```

This probability tells us **how likely the email is spam**.

```python
# Basic Logistic Regression with Iris Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Load iris dataset
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)  # Binary: Virginica vs Others

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Sample predictions: {y_pred[:5]}")
print(f"Sample probabilities: {y_prob[:5]}")
```

---

## 5. Cost Function for Logistic Regression

### **The Cost Function**
```
J(θ) = -(1/m) Σ [y⁽ⁱ⁾ log(hθ(x⁽ⁱ⁾)) + (1 - y⁽ⁱ⁾) log(1 - hθ(x⁽ⁱ⁾))]
```

**Where:**
- `m` = number of training examples
- `y⁽ⁱ⁾` = actual label for i-th example (0 or 1)
- `hθ(x⁽ⁱ⁾)` = predicted probability for i-th example

### **Interpretation**
- **y⁽ⁱ⁾ log(hθ(x⁽ⁱ⁾))**: Cost when actual label is 1
- **(1 - y⁽ⁱ⁾) log(1 - hθ(x⁽ⁱ⁾))**: Cost when actual label is 0

**Goal**: Minimize this cost to make predictions as close to actual labels as possible.

---

## 6. Training: Gradient Descent

### **Purpose**
Minimize cost function using **Gradient Descent** optimization.

### **Process**
- Start with **initial weights and biases**
- **Iteratively adjust** them to reduce cost

### **Update Rule**
```
θⱼ := θⱼ - α (∂J(θ)/∂θⱼ)
```

**Where:**
- `θⱼ` = each parameter (weights wⱼ and bias b)
- `α` = **learning rate** (controls update step size)

```python
# Custom implementation of gradient descent
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow

def cost_function(X, y, theta):
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    cost = -(1/m) * (y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    
    for i in range(iterations):
        z = X.dot(theta)
        h = sigmoid(z)
        gradient = (1/m) * X.T.dot(h - y)
        theta = theta - alpha * gradient
        cost = cost_function(X, y, theta)
        costs.append(cost)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.6f}")
    
    return theta, costs

# Example usage with iris data
from sklearn.preprocessing import StandardScaler

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])  # Add bias term

# Initialize parameters
theta = np.random.normal(0, 0.01, X_scaled.shape[1])

# Train
theta_final, costs = gradient_descent(X_scaled, y_train, theta, alpha=0.01, iterations=1000)

print(f"Final parameters: {theta_final}")
```

---

## 7. Evaluation Metrics for Classification

### **Confusion Matrix**

A **2x2 table** showing actual vs predicted classifications:

```
                 Predicted
              Positive  Negative
Actual Positive   TP      FN
       Negative   FP      TN
```

**Where:**
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative  
- **FP (False Positive)**: Wrongly predicted positive (**Type I Error**)
- **FN (False Negative)**: Wrongly predicted negative (**Type II Error**)

### **Medical Example: Cancer Diagnosis**

**Scenario**: 100 patients tested
- **99 are healthy** (non-malignant)
- **1 has cancer** (malignant)

**Naive Model**: Always predicts "healthy"
- **Accuracy**: 99/100 = 99%
- **Problem**: Missed the cancer patient! (Type II Error)

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Create example: 100 patients (99 healthy, 1 cancer)
y_true = [0]*99 + [1]*1
y_pred_naive = [0]*100  # Naive model: always predict healthy

# Confusion matrix
cm_naive = confusion_matrix(y_true, y_pred_naive)
print("Naive Model Confusion Matrix:")
print(cm_naive)
print(f"Accuracy: {(cm_naive[0,0] + cm_naive[1,1])/100:.1%}")
print(f"Missed cancer cases: {cm_naive[1,0]}")

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_naive, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Naive Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### **Key Metrics Explained**

#### **1. Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **When useful**: Balanced datasets
- **Problem**: Misleading with imbalanced data

#### **2. Precision**
```
Precision = TP / (TP + FP)
```
- **Question**: Of all positive predictions, how many were correct?
- **Focus**: Avoiding **false alarms**

#### **3. Recall (Sensitivity)**
```
Recall = TP / (TP + FN)
```
- **Question**: Of all actual positives, how many did we catch?
- **Focus**: Avoiding **missed cases**

#### **4. Specificity**
```
Specificity = TN / (TN + FP)
```
- **Question**: Of all actual negatives, how many were correctly identified?

#### **5. F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Balance** between precision and recall

```python
# Detailed evaluation with iris data
from sklearn.metrics import precision_score, recall_score, f1_score

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Metrics:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))
```

### **Threshold Adjustment for Different Needs**

**Default threshold**: 0.5
- If **p(positive) ≥ 0.5** → Predict positive
- If **p(positive) < 0.5** → Predict negative

#### **Scenario 1: Medical Screening (Reduce Type II Error)**
- **Goal**: Don't miss any cancer cases
- **Solution**: **Lower threshold** (e.g., 0.3)
- **Result**: Higher recall, more false alarms

#### **Scenario 2: Spam Detection (Reduce Type I Error)**  
- **Goal**: Don't block important emails
- **Solution**: **Higher threshold** (e.g., 0.7)
- **Result**: Higher precision, might miss some spam

```python
# Threshold adjustment example
thresholds = [0.3, 0.5, 0.7]
y_proba = model.predict_proba(X_test)[:, 1]

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    
    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
```

### **ROC Curve and AUC**

**ROC (Receiver Operating Characteristic)**: Plot of True Positive Rate vs False Positive Rate

```python
from sklearn.metrics import roc_curve, auc

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

---

## 8. Complete Practical Example

```python
# Complete workflow with iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)  # Binary classification: Virginica vs Others

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("=== MODEL EVALUATION ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature importance
feature_names = iris.feature_names
importance = abs(model.coef_[0])
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importance)
plt.title('Feature Importance (|Coefficients|)')
plt.xlabel('Importance')
plt.show()

print("\n=== FEATURE COEFFICIENTS ===")
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"{name}: {coef:.3f}")
```

---

## 9. Key Takeaways

### **When to Use Logistic Regression**
- **Binary classification** problems
- Need **interpretable** results
- **Linear relationship** between features and log-odds
- **Fast training** and prediction needed

### **Advantages**
- **Simple and interpretable**
- **No tuning** of hyperparameters
- **Probability outputs**
- **Fast** training and prediction

### **Limitations**  
- Assumes **linear relationship**
- Sensitive to **outliers**
- Requires **large sample** sizes
- **Feature scaling** needed

### **Best Practices**
- **Scale features** before training
- **Check for multicollinearity**
- **Choose appropriate threshold** based on business needs
- **Use cross-validation** for model selection
- **Consider regularization** (Ridge/Lasso) for high-dimensional data

---

## 10. Next Steps

1. **Practice** with different datasets
2. **Experiment** with threshold tuning
3. **Learn regularization** (L1/L2)
4. **Explore** multiclass classification
5. **Study** advanced evaluation metrics
6. **Compare** with other algorithms

---

*This completes our comprehensive journey through Logistic Regression! Remember: the key is understanding when and why to use it, not just how to implement it.*