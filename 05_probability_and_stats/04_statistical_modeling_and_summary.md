## Statistical Modeling - Making Predictions

### From Correlation to Causation

**Why Statistical Modeling?**
• Understanding relationships between variables
• Making predictions
• Identifying important factors
• Providing interpretable results

**Correlation vs Causation:**
• Correlation: Variables move together
• Causation: One variable influences another
• Models help us understand both

---

### Simple Linear Regression

**The Basic Idea:**
Fit a straight line through data points to model relationship between X and Y

**The Equation:**
Y = β₀ + β₁X + ε

Where:
• β₀ = intercept (Y when X = 0)
• β₁ = slope (change in Y for unit change in X)
• ε = error term

**Interpreting Coefficients:**
"For every one unit increase in X, Y increases by β₁ units, on average"

**Example:**
House Price = 50,000 + 100 × Square_Feet
• Base price: $50,000
• Each additional square foot adds $100 to price

---

### Checking Model Assumptions

**Key Assumptions:**
• **Linearity**: Relationship is actually linear
• **Independence**: Observations don't influence each other
• **Normality**: Errors are normally distributed
• **Homoscedasticity**: Constant variance of errors

**Residual Analysis:**
• Residuals = Actual - Predicted values
• Plot residuals vs predicted values
• Look for patterns that violate assumptions

**What Good Residuals Look Like:**
• Randomly scattered around zero
• No clear patterns or trends
• Roughly constant spread

---

### Statistical Significance in Regression

**Testing Coefficient Significance:**
H₀: β₁ = 0 (no relationship)
H₁: β₁ ≠ 0 (significant relationship)

**The t-statistic:**
t = (β̂₁ - 0) / SE(β̂₁)

**Interpretation:**
• Large |t| and small p-value → significant relationship
• Coefficient is "statistically significant"
• Variable is a meaningful predictor

**R-squared:**
• Proportion of variance in Y explained by X
• Ranges from 0 to 1
• Higher is better (but don't chase it blindly)

---

### Logistic Regression - Modeling Probabilities

**When to Use Logistic Regression:**
• Outcome is binary (yes/no, success/failure)
• Want to model probability of success
• Examples: email spam, customer churn, medical diagnosis

**The Logistic Function:**
P(Y=1) = e^(β₀ + β₁X) / (1 + e^(β₀ + β₁X))

**Key Properties:**
• Output always between 0 and 1
• S-shaped curve
• Linear relationship with log-odds

**Interpreting Coefficients:**
• e^β₁ = odds ratio
• "One unit increase in X multiplies odds by e^β₁"
• Positive β₁ → increases probability
• Negative β₁ → decreases probability

**Example:**
Churn Model: Log-odds(Churn) = -2 + 0.5×Complaints + 1.2×MonthsSinceLastPurchase
• e^0.5 = 1.65: Each complaint increases churn odds by 65%
• e^1.2 = 3.32: Each month since purchase increases odds by 232%

---

## Course Summary and Next Steps

### Key Takeaways

**Probability Foundations:**
• Uncertainty is everywhere in AI
• Bayes' rule is fundamental to machine learning
• Understanding distributions helps in model selection

**Data Analysis Skills:**
• EDA prevents costly modeling mistakes
• Visualization reveals insights that numbers alone cannot
• Always check your assumptions

**Statistical Inference:**
• Central Limit Theorem enables all statistical inference
• Confidence intervals quantify uncertainty
• Hypothesis testing helps make decisions under uncertainty

**Modeling for Understanding:**
• Start simple before going complex
• Statistical models provide interpretability
• Always validate your model assumptions

### Preparing for Advanced Topics

This foundation prepares you for:
• Machine Learning algorithms
• Deep Learning concepts
• Advanced statistical methods
• Experimental design and causal inference

### Final Advice

• Practice with real datasets
• Always visualize your data first
• Question your results - statistics can be misleading
• Remember: the goal is insight, not just prediction