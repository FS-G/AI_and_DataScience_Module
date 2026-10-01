# Statistics for AI: Complete Course Lectures

## Module 1: The Foundations of Probability

### Lecture 1.1: Why Probability Matters in AI

**Learning Objectives:**
- Understand why uncertainty is everywhere in AI
- Connect probability to real AI applications

**Key Points:**
• AI systems deal with uncertainty constantly
  - Will it rain tomorrow? (Weather prediction)
  - Is this email spam? (Classification)
  - What movie will you like? (Recommendation systems)

• Probability gives us a mathematical language for uncertainty
  - Instead of saying "maybe" or "probably not"
  - We can say "30% chance" or "0.7 probability"

**Real-World Example:**
A medical AI says a patient has a 85% chance of having diabetes based on symptoms. This probability helps doctors make informed decisions.

---

### Lecture 1.2: Basic Probability Concepts

**Sample Space and Events**
• **Sample Space (S)**: All possible outcomes
  - Coin flip: S = {Heads, Tails}
  - Dice roll: S = {1, 2, 3, 4, 5, 6}

• **Event**: A subset of outcomes we're interested in
  - Getting heads in coin flip
  - Rolling an even number on dice

**Probability Rules (Axioms)**
• Probability is always between 0 and 1
  - P(Event) ≥ 0
  - P(Sample Space) = 1
  - P(Impossible Event) = 0

**Simple Example:**
Rolling a fair dice:
- P(rolling 3) = 1/6 ≈ 0.167
- P(rolling even number) = P(2, 4, or 6) = 3/6 = 0.5

---

### Lecture 1.3: Combining Probabilities

**Joint Probability P(A and B)**
• Probability that both events happen
• Example: P(Rain AND Cold) = probability of rainy and cold weather

**Marginal Probability P(A)**
• Probability of one event, regardless of others
• Example: P(Rain) = probability of rain, whether cold or warm

**Simple Calculation:**
If we flip two coins:
- P(Both Heads) = P(H₁ and H₂) = 0.5 × 0.5 = 0.25
- P(At least one Head) = 1 - P(Both Tails) = 1 - 0.25 = 0.75

---

### Lecture 1.4: Conditional Probability - The Game Changer

**What is Conditional Probability?**
Think of it as "probability with extra information"

• P(A|B) = Probability of A happening, **knowing that B already happened**
• Read as "Probability of A given B"

**Everyday Example - Traffic and Being Late:**
• P(Late for Work) might be 10% on a normal day
• P(Late for Work | Heavy Traffic) might be 60%
• Same outcome, but extra information changes everything!

**Another Relatable Example - Netflix and Mood:**
• P(Watch Comedy) = 30% (general preference)
• P(Watch Comedy | Had Bad Day) = 80% (comfort viewing)
• P(Watch Comedy | Celebrating) = 60% (feel-good content)

**The Key Insight:**
New information completely changes probabilities. This is why AI systems ask for context!

**Simple Formula:**
P(A|B) = P(A and B) / P(B)

**In Plain English:**
"How often A and B happen together" ÷ "How often B happens"

**Visual Thinking:**
Imagine all the days B happens. Of those days, what fraction also has A?

---

### Lecture 1.5: Bayes' Rule - The Heart of AI Learning

**The Revolutionary Idea:**
Bayes' rule lets us "flip" probabilities and learn from evidence, just like humans do!

**The Setup - Two Questions:**
1. **Forward:** If I have the flu, what's the chance I have a fever?
2. **Reverse:** If I have a fever, what's the chance I have the flu?

These are completely different questions with different answers!

**Bayes' Formula:**
P(A|B) = P(B|A) × P(A) / P(B)

**In Everyday Language:**
P(Flu|Fever) = P(Fever|Flu) × P(Flu) / P(Fever)

**Breaking Down Each Part:**
• **P(Fever|Flu)**: How often do flu patients get fever? (Easy to measure)
• **P(Flu)**: How common is flu in general? (Base rate)
• **P(Fever)**: How often do people get fever? (From any cause)
• **P(Flu|Fever)**: What we want to know!


**Another Example:**  
P(Passed|Studied) = P(Studied|Passed) × P(Passed) / P(Studied)  

**Breaking Down Each Part:**  
• **P(Studied|Passed)**: Among students who passed, how many actually studied? (Easy to measure)  
• **P(Passed)**: How common is it to pass in general? (Base rate)  
• **P(Studied)**: How many students study at all? (Easy to find out)  
• **P(Passed|Studied)**: What we want to know!


---

**Real-World Example - Email Spam Detection:**

**Setup:**
• 2% of all emails are actually spam
• Your spam filter is 95% accurate
• When an email is flagged as spam, what's the chance it's actually spam?

**Your Gut Feeling:** Probably 95% (filter accuracy)

**Let's Think Through This:**
• Out of 1000 emails: 20 are spam, 980 are legitimate
• Filter correctly identifies: 19 spam emails (95% of 20)
• Filter incorrectly flags: 49 legitimate emails as spam (5% of 980)
• Total flagged as spam: 19 + 49 = 68 emails

**The Reality:** Only 19 out of 68 flagged emails are actually spam = 28%

**Why This Matters for AI:**
Even very accurate AI systems can have surprising real-world performance when dealing with rare events!

---

**Key Assumptions of Bayesian Thinking:**

**1. Prior Knowledge Matters**
• We always start with some belief (prior probability)
• This could be from data, experience, or reasonable assumptions
• If you have no information, use "uninformative priors"

**2. Evidence Updates Beliefs**
• New information should change your mind
• Strong evidence = big updates
• Weak evidence = small updates

**3. All Information is Probabilistic**
• Nothing is 100% certain (except math)
• Tests can be wrong, witnesses can be mistaken
• We work with degrees of belief

**4. Independence Assumptions**
• Often assume events don't influence each other
• Example: Assumes your test result doesn't influence whether others have the disease
• This can be violated in real life (epidemics, genetic factors)

---

**Why This Matters for AI:**

**Spam Filters:**
• P(Spam | Contains "FREE MONEY") uses Bayes' rule
• Learns from millions of examples
• Updates beliefs as new emails arrive

**Medical Diagnosis:**
• Combines multiple symptoms and test results
• Updates probability as more information comes in
• Helps doctors avoid overconfidence in single tests

**Recommendation Systems:**
• P(You'll like Movie X | You liked Movies A, B, C)
• Learns from your behavior and others like you
• Updates recommendations as you rate more movies

**The Big Lesson:**
Bayes' rule is how rational thinking works - start with what you know, update with new evidence, and always consider base rates!

---

## Module 2: Common Probability Distributions

### Lecture 2.1: Random Variables - Turning Outcomes into Numbers

**What is a Random Variable?**
• A function that assigns numbers to outcomes
• Bridges the gap between events and mathematics

**Types:**
• **Discrete**: Can count the values (1, 2, 3...)
  - Number of emails received per day
  - Number of website clicks

• **Continuous**: Can measure on a scale (any real number)
  - Temperature
  - Stock prices
  - Time between events

---

### Lecture 2.2: Essential Probability Functions

**Probability Mass Function (PMF) - For Discrete Variables**
• Shows probability of each specific value
• **Example - Flipping 3 coins, counting heads:**
  - P(X=0) = 1/8 (TTT)
  - P(X=1) = 3/8 (HTT, THT, TTH)  
  - P(X=2) = 3/8 (HHT, HTH, THH)
  - P(X=3) = 1/8 (HHH)

**Probability Density Function (PDF) - For Continuous Variables**
• Shows relative likelihood of values (height of curve)
• **Example - Height of students:**
  - Curve is tallest around 5'6" (most common height)
  - Gets shorter toward 4'0" and 7'0" (rare heights)
  - Total area under curve = 100%

**Cumulative Distribution Function (CDF)**
• P(X ≤ x) = probability of getting value x or less
• **Example - Test scores (out of 100):**
  - CDF(70) = 0.60 means 60% of students scored 70 or below
  - CDF(90) = 0.90 means 90% of students scored 90 or below
  - Always increases: if you scored higher, more students are below you

---

### Lecture 2.3: The Discrete Distribution Family

![Discrete Distributions Overview](https://blogger.googleusercontent.com/img/a/AVvXsEhx17znSE17sn4oPqq3bXvxw7_Df0zGsN9imB7AK6_IDA5nrBq-A1aIG5q03zK_CNLlGRawxBQIWUKPQU4vh3U34YcONBFU75kqfbphhG_WVTcy52vkutvsf5AO8X_f-KCd_6C1uH27PDXMFbXiKWWb2h_mROTgR5KPsiHhOGxG3rd8vrRyWFdyEdMp)


![Exponential Disctibution](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Exponential_distribution_pdf_-_public_domain.svg/488px-Exponential_distribution_pdf_-_public_domain.svg.png)

**Bernoulli Distribution - The Yes/No Distribution**
• Models single trial with two outcomes
• Examples:
  - Click/No Click on ad
  - Spam/Not Spam email
  - Success/Failure

• Parameter: p = probability of success
• P(X=1) = p, P(X=0) = 1-p

**Binomial Distribution - Multiple Bernoulli Trials**
• Models number of successes in n trials
• Examples:
  - Number of sales out of 10 customer visits
  - Number of correct predictions in 100 test cases

• Parameters: n (trials), p (success probability)
• Mean = np, Variance = np(1-p)

**Poisson Distribution - Counting Rare Events**
• Models number of events in fixed time period
• Examples:
  - Number of website crashes per day
  - Number of customer service calls per hour

• Parameter: λ (average rate)
• Mean = Variance = λ

---

### Lecture 2.4: The Continuous Distribution Family

**Uniform Distribution - Equal Chances**
• All values in range equally likely
• Examples:
  - Random number generator
  - Initial neural network weights

• Perfect for modeling "complete uncertainty"

**Normal Distribution - The Bell Curve**
• Most important distribution in statistics
• Examples:
  - Heights of people
  - Measurement errors
  - Many natural phenomena

• Parameters: μ (mean), σ (standard deviation)
• Bell-shaped, symmetric
• 68-95-99.7 rule

**Exponential Distribution - Waiting Times**
• Models time between events
• Examples:
  - Time between customer arrivals
  - Time until system failure

• Parameter: λ (rate)
• "Memoryless" property

---

## Module 3: Exploratory Data Analysis (EDA)

### Lecture 3.1: Why EDA is Critical for AI

**The Foundation of Good AI**
• "Garbage in, garbage out" - bad data = bad models
• EDA helps us understand our data before building models
• Prevents costly mistakes and wrong conclusions

**What EDA Reveals:**
• Data quality issues (missing values, outliers)
• Hidden patterns and relationships
• Assumptions we need to check
• Ideas for feature engineering

---

### Lecture 3.2: Measures of Central Tendency

**Mean - The Average**
• Sum of all values / number of values
• Sensitive to outliers
• Best for symmetric distributions

**Median - The Middle Value**
• 50th percentile when data is sorted
• Robust to outliers
• Better for skewed distributions

**Mode - The Most Common Value**
• Value that appears most frequently
• Can have multiple modes
• Useful for categorical data

**Example - House Prices:**
Houses sold: $200K, $250K, $300K, $320K, $2M
- Mean = $614K (pulled up by mansion)
- Median = $300K (better representative)
- Mode = None (all different)

**Key Insight:** When mean >> median, data is right-skewed (has high outliers)

---

### Lecture 3.3: Measures of Spread

**Variance and Standard Deviation**
• How much data varies around the mean
• **Variance = Average of squared differences from mean**
  - Formula: Var = Σ(x - mean)² / n
• **Standard Deviation = √Variance**
  - Same units as original data
• **Simple Example - Test Scores:** [70, 80, 90]
  - Mean = 80
  - Variance = [(70-80)² + (80-80)² + (90-80)²] / 3 = [100 + 0 + 100] / 3 = 67
  - Standard Deviation = √67 = 8.2 points

**Interquartile Range (IQR)**
• 75th percentile - 25th percentile
• Robust to outliers
• Contains middle 50% of data
• **Example - Daily Coffee Sales:** [12, 15, 18, 20, 22, 25, 28, 30, 35, 100]
  - Sort data (already sorted)
  - Q1 (25th percentile) = 18 cups
  - Q3 (75th percentile) = 30 cups
  - IQR = 30 - 18 = 12 cups
  - Notice: The outlier (100 cups) doesn't affect IQR!

**Range**
• Maximum - Minimum
• Very sensitive to outliers
• Simple but not very informative

**Outlier Detection Rule:**
• Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
• Simple rule for identifying unusual data points

---

### Lecture 3.4: Understanding Distribution Shapes

**Skewness - Is it Symmetric?**
• Positive skew: Long tail to the right (income, house prices)
• Negative skew: Long tail to the left (exam scores in easy test)
• Zero skew: Symmetric (height, temperature)

**Kurtosis - How "Peaky" is it?**
• High kurtosis: Sharp peak, heavy tails
• Low kurtosis: Flat top, light tails
• Normal distribution has kurtosis = 3

**Why This Matters for AI:**
• Many algorithms assume normal distributions
• Skewed data might need transformation (log transform)
• Kurtosis affects outlier sensitivity

---

### Lecture 3.5: Essential Visualizations

**Univariate Analysis - Exploring Single Variables**

**Histograms:**
• Shows distribution of one continuous variable
• Bins group data into ranges
• Look for: shape (normal, skewed), outliers, multiple peaks
• **Tip:** Try different bin widths - too few bins hide patterns, too many create noise

**Box Plots (Single Variable):**
• Shows five-number summary: min, Q1, median, Q3, max
• Outliers appear as individual points beyond "whiskers"
• Quickly reveals skewness and spread

**Bar Charts:**
• For categorical variables (colors, brands, regions)
• Height shows frequency or percentage
• Easy comparison between categories

**Density Plots:**
• Smooth version of histogram
• Better for comparing multiple groups on same plot
• Shows probability density rather than counts

---

**Bivariate Analysis - Exploring Relationships Between Two Variables**

**Scatter Plots:**
• Two continuous variables (X vs Y)
• Each point represents one observation
• Look for: linear/curved relationships, outliers, clusters
• Foundation for correlation and regression analysis

**Side-by-Side Box Plots:**
• One continuous variable across categories
• Compare distributions between groups
• Quickly shows which group has higher median, more variation

**Cross-Tabulation Tables:**
• Two categorical variables
• Shows frequency counts in each combination
• Foundation for chi-square tests

**Correlation Heatmaps:**
• Shows correlation coefficients between multiple variables
• Color intensity indicates strength of relationship
• Quick way to spot highly correlated pairs

---

**Multivariate Analysis - Exploring Multiple Variables Together**

**Pair Plots (Scatter Plot Matrix):**
• Shows scatter plots for every pair of variables
• Diagonal usually shows histograms of each variable
• Great for initial exploration of datasets



---

**Special Purpose Plots**

**QQ Plots - Testing Normality:**
• Compares your data to theoretical normal distribution
• Points on straight diagonal line = data is normal
• S-curves indicate skewness
• Curved ends indicate heavy/light tails
• Essential before using methods that assume normality

---

## Module 4: Central Limit Theorem - The Magic Behind Statistics

### Lecture 4.1: Population vs Sample - The Fundamental Distinction

**Population**
• All possible data points we care about
• Usually impossible to measure completely
• Has true parameters (μ, σ)

**Sample**
• Subset of population we actually observe
• What we use to make inferences
• Has sample statistics (x̄, s)

**Examples:**
• Population: All customers who might buy our product
• Sample: 1000 customers we surveyed

**Key Challenge:** How can we trust conclusions from a sample?

---

### Lecture 4.2: Law of Large Numbers - Building Intuition

**The Simple Idea:**
As sample size increases, sample mean gets closer to true population mean

**Coin Flip Example:**
• Flip 10 times: might get 70% heads (7/10)
• Flip 1000 times: likely close to 50% heads
• Flip 1 million times: very close to 50% heads

**Why This Matters for AI:**
• Gives us confidence that training data represents reality
• Justifies using sample performance to estimate true performance
• Explains why "more data usually helps"

---

### Lecture 4.3: Central Limit Theorem - The Miracle of Statistics

**The Amazing Result:**
No matter what the original population looks like, the distribution of sample means will be approximately normal if sample size is large enough (usually n ≥ 30)

**Key Points:**
• Works for ANY population distribution (uniform, exponential, bimodal...)
• Sample means have less variability than individual observations
• Standard error = σ/√n (gets smaller as n increases)

**Simple Example:**
• Roll dice (uniform distribution from 1-6)
• Take samples of 30 rolls, calculate mean of each sample
• Plot histogram of these sample means
• Result: Beautiful bell curve centered at 3.5!

**Why This is Magical:**
• Enables all of statistical inference
• Explains why we can make probability statements about estimates
• Foundation of confidence intervals and hypothesis testing

---

## Module 5: Confidence Intervals - How Sure Are We?

### Lecture 5.1: From Point to Interval Estimates

**Point Estimate - A Single Number**
• Sample mean = 85% accuracy
• But how precise is this estimate?

**Interval Estimate - A Range of Plausible Values**
• Example: "Average student height is 5'6" ± 2 inches"
• Instead of just saying 5'6", we say "between 5'4" and 5'8""
• Acknowledges uncertainty in our estimate

**Why Intervals Matter in AI:**
• Model A: 90% ± 5% accuracy
• Model B: 85% ± 1% accuracy
• Which is better? Depends on the application!

---

### Lecture 5.2: Understanding Confidence Intervals

**The Formula:**
Point Estimate ± (Critical Value × Standard Error)

**Components:**
• **Point Estimate**: Our best guess (sample mean)
• **Critical Value**: From normal distribution (1.96 for 95% CI)
• **Standard Error**: Standard deviation of sampling distribution

**Correct Interpretation:**
"If we repeated this study 100 times, about 95 of the confidence intervals would contain the true population parameter"

**Common Misinterpretation:**
"There's a 95% chance the true value is in this interval" (Wrong!)

---

### Lecture 5.3: Constructing Confidence Intervals

**Understanding z vs t Distributions**

![z vs t distribution](https://cdn.prod.website-files.com/6634a8f8dd9b2a63c9e6be83/669d64959d5970e08c48ad1c_360214.image0.jpeg)

**z Distribution (Standard Normal):**
• Use when population standard deviation (σ) is **known**
• Bell-shaped, mean=0, std dev=1
• Fixed shape, same critical values always
• Example: z = 1.96 for 95% CI

**t Distribution:**
• Use when population standard deviation (σ) is **unknown** (most real cases!)
• Similar to z but with "fatter tails" (more uncertainty)
• Shape depends on degrees of freedom (df = n-1)
• As sample size increases, t approaches z distribution
• For n>30: t ≈ z (practically the same)

**When to Use Which:**

| Situation | Distribution | Formula |
|-----------|-------------|---------|
| σ known (rare) | z | x̄ ± z_(α/2) × (σ/√n) |
| σ unknown, n≤30 | t | x̄ ± t_(α/2,df) × (s/√n) |
| σ unknown, n>30 | z or t | x̄ ± z_(α/2) × (s/√n) |

**For a Proportion (always use z):**
p̂ ± z_(α/2) × √(p̂(1-p̂)/n)

**Worked Example - Customer Satisfaction (Using Mean Formula):**
• Sample: 100 customers
• Sample mean satisfaction = 7.2 (out of 10)
• Sample standard deviation = 1.5
• Want 95% confidence interval

**Step-by-Step Calculation:**
• **Formula Used:** x̄ ± z_(α/2) × (s/√n) (unknown σ, large sample)
• **Critical Value:** z_(0.025) = 1.96 (for 95% CI)
• **Standard Error:** s/√n = 1.5/√100 = 1.5/10 = 0.15
• **Margin of Error:** 1.96 × 0.15 = 0.294
• **Final CI:** 7.2 ± 0.294 = [6.91, 7.49]

**Interpretation:** We're 95% confident the true average customer satisfaction is between 6.91 and 7.49

---

### Lecture 5.4: Bootstrap - A Modern Approach

**The Bootstrap Idea:**
• Resample from your original sample (with replacement)
• Calculate statistic for each resample
• Use distribution of these statistics to create CI

**Why Bootstrap is Powerful:**
• Works for any statistic (median, correlation, etc.)
• No complex formulas needed
• Very intuitive

**Simple Bootstrap Example:**
Original sample: [2, 4, 6, 8, 10]
Bootstrap sample 1: [2, 6, 6, 10, 4] → mean = 5.6
Bootstrap sample 2: [8, 8, 2, 10, 6] → mean = 6.8
... (repeat 1000 times)
95% CI = 2.5th and 97.5th percentiles of bootstrap means

---

## Module 6: Hypothesis Testing - Is This Effect Real?

### Lecture 6.1: The Logic of Hypothesis Testing

**The Scientific Method in Statistics:**
• Start with a claim to test
• Assume the opposite (null hypothesis)
• Collect evidence
• Decide if evidence is strong enough to reject the null

**Key Components:**
• **Null Hypothesis (H₀)**: "No effect" or "no difference"
• **Alternative Hypothesis (H₁)**: What we want to prove
• **Test Statistic**: Measures how far our data is from H₀
• **P-value**: Probability of seeing our result if H₀ is true

---

### Lecture 6.2: Understanding P-values

**What is a P-value?**
Probability of observing a test statistic as extreme as (or more extreme than) what we actually observed, assuming the null hypothesis is true

**Common Misinterpretations:**
❌ "Probability that null hypothesis is true"
❌ "Probability of making a mistake"
✅ "Probability of our data, given null hypothesis is true"

**Decision Rule:**
• If p-value < α (usually 0.05): Reject null hypothesis
• If p-value ≥ α: Fail to reject null hypothesis

**Example:**
H₀: New website design doesn't improve conversion rate
We observe p-value = 0.03
Conclusion: If the new design really had no effect, we'd only see results this extreme 3% of the time. This is unlikely, so we reject H₀.

---

### Lecture 6.3: Types of Errors

**Type I Error (False Positive)**
• Rejecting H₀ when it's actually true
• "Crying wolf" - seeing an effect that isn't there
• Probability = α (significance level)

**Type II Error (False Negative)**
• Failing to reject H₀ when it's actually false
• Missing a real effect
• Probability = β



**Balancing Act:**
• Lower α → Lower Type I error, but higher Type II error
• Like adjusting sensitivity of a medical test

---

### Lecture 6.4: Common Statistical Tests

**One-Sample t-test**
• Tests if sample mean differs from known value
• H₀: μ = μ₀
• Example: Is average customer rating different from 4.0?

**Two-Sample t-test**
• Tests if two group means are different
• H₀: μ₁ = μ₂
• Example: Do men and women have different average spending?

**Proportion Test**
• Tests if sample proportion differs from known value
• H₀: p = p₀
• Example: Is click-through rate different from 5%?

---

### Lecture 6.5: A/B Testing - Statistics in Action

**The Setup:**
• Group A: Current website (control)
• Group B: New website (treatment)
• Question: Is conversion rate significantly different?

**The Test:**
H₀: p_B = p_A (no difference in conversion rates)
H₁: p_B ≠ p_A (there is a difference)

**Example Calculation:**
• Group A: 100 conversions out of 2000 visitors (5%)
• Group B: 130 conversions out of 2000 visitors (6.5%)
• Two-proportion z-test gives p-value = 0.018
• Conclusion: Reject H₀, new design performs better

**Practical Considerations:**
• Sample size planning
• Multiple testing corrections
• Business significance vs statistical significance

---

## Module 7: Statistical Modeling - Making Predictions

### Lecture 7.1: From Correlation to Causation

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

### Lecture 7.2: Simple Linear Regression

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

### Lecture 7.3: Checking Model Assumptions

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

### Lecture 7.4: Statistical Significance in Regression

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

### Lecture 7.5: Logistic Regression - Modeling Probabilities

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