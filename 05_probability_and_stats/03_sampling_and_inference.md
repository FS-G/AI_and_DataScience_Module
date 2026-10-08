## Central Limit Theorem - The Magic Behind Statistics

### Population vs Sample - The Fundamental Distinction

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

### Law of Large Numbers - Building Intuition

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

### Central Limit Theorem - The Miracle of Statistics

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

## Confidence Intervals - How Sure Are We?

### From Point to Interval Estimates

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

### Understanding Confidence Intervals

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

### Constructing Confidence Intervals

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

### Bootstrap - A Modern Approach

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

## Hypothesis Testing - Is This Effect Real?

### The Logic of Hypothesis Testing

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

### Understanding P-values

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

### Types of Errors

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

### Common Statistical Tests

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

### A/B Testing - Statistics in Action

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

