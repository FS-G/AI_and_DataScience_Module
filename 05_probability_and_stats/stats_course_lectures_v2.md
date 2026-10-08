**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*

---

# Statistics for AI — Version 2

## Course goal

Statistics helps us make sensible decisions when data is incomplete or uncertain. In this course, we will use small examples and Python to describe data, compare groups, and make careful conclusions.

We will use `pandas`, `matplotlib`, and `scipy` where helpful. The focus is on understanding what a result means, not memorizing formulas.

```python
import pandas as pd
import matplotlib.pyplot as plt
```

## 1. Start with a question and a dataset

Before calculating anything, ask a clear question. Examples:

- What is the typical order value?
- Do customers who receive a discount buy more?
- Did a new website design increase sign-ups?

A dataset is a table: each row is one observation, and each column is a feature about it. For example, a sales table might have one row per order and columns for date, product, quantity, and sales amount.

### Practical check

Load a dataset and inspect its first rows, column names, and size. This helps you learn what one row represents and what questions the data can answer.

```python
df = pd.read_csv("03_python_for_data_analysis_and_visualization/data/sales.csv")
print(df.head())
print(df.shape)
print(df.columns.tolist())
```

If you are running the code from another folder, adjust the file path. Check for missing values and repeated rows before trusting the results.

```python
print(df.isna().sum())
print(df.duplicated().sum())
```

**Takeaway:** Understand where the data came from and what each row means before summarizing it.

## 2. Describe what you see

Descriptive statistics summarize the data you already have. They do not automatically explain why something happened.

### Counts and proportions

For categories, count how many records fall into each group. A proportion is a count divided by the total.

```python
print(df["Product"].value_counts())
print(df["Product"].value_counts(normalize=True).round(2))
```

Use a bar chart to compare categories. Ask: Which category is most common? Are some groups very small?

### Typical values: mean and median

The **mean** is the arithmetic average. The **median** is the middle value after sorting. A very large or small value can pull the mean, so compare both.

```python
amount = df["Sales"].dropna()
print("Mean:", amount.mean())
print("Median:", amount.median())
```

If the mean is much higher than the median, a few large sales may be pulling up the average. In that case, the median may better describe a typical sale.

### Spread and unusual values

The range shows the distance from the smallest to largest value. The standard deviation describes typical distance from the mean. The interquartile range (IQR) describes the spread of the middle half and is less affected by extreme values.

```python
print(amount.describe())
print("IQR:", amount.quantile(0.75) - amount.quantile(0.25))
amount.plot(kind="hist", title="Sales amounts", xlabel="Sales")
plt.show()
```

A histogram shows the shape of numeric data. A box plot can help spot unusually high or low values. An unusual value may be an error, or it may be a real and important observation. Check before removing it.

### Practical exercise

Choose one numeric column. Report its count, mean, median, and IQR. Draw a histogram and explain in one sentence what a typical value looks like and whether there are unusual values.

## 3. Understand chance with simple examples

A **probability** is a number from 0 to 1. Zero means an event cannot happen; one means it is certain. We often show probabilities as percentages.

For a fair coin, the chance of heads is 0.5, or 50%. If we flip it 10 times, we should not expect exactly 5 heads every time. Random results vary.

### Conditional probability

A conditional probability answers a question with extra information: “Among the records that meet condition B, how many also meet condition A?”

For example, calculate the share of orders over $100 among orders in each market:

```python
large_order = df["Sales"] > 100
print(df.groupby("Market")[large_order].size())
```

A clearer way to calculate the share is to create a yes/no column and take its average (True counts as 1):

```python
df["LargeOrder"] = (df["Sales"] > 100).astype(int)
print(df.groupby("Market")["LargeOrder"].mean().round(2))
```

**Takeaway:** Always name the group you are comparing. “Among orders in this market” is different from “among all orders.”

### Bayes’ rule in plain language

Bayes’ rule updates a probability when new evidence arrives. A spam filter, for example, should consider both how often spam occurs and how often a word appears in spam messages. A clue alone does not tell us the answer; the starting rate matters too.

You do not need to calculate Bayes’ rule by hand in every example. Focus on this question: “Given what I observed, how should I update my estimate?”

## 4. Check whether your sample is fair

A **population** is the full group we want to understand. A **sample** is the part we actually observe. We use samples because measuring everyone is often too slow or costly.

A sample can mislead us if it leaves out important people or cases. This is called **sampling bias**. For example, asking only online customers about a shop may miss customers who do not shop online. A larger biased sample can still give a wrong picture.

### Practical activity

Imagine asking 100 people about a new app, but recruiting all 100 from its fan group. Discuss:

1. Who is missing from the sample?
2. What result might be overstated?
3. How could we recruit a more balanced sample?

When working with a dataset, check who or what is included, when the data was collected, and whether records are missing for a reason.

## 5. Common patterns in data

A **distribution** shows which values are common and which are rare. A histogram is a practical way to see it.

- **Binary outcome:** two possibilities, such as clicked or did not click.
- **Counts:** number of events, such as orders per day.
- **Measurements:** values such as delivery time or sales amount.

Many measurements have a roughly bell-shaped pattern, but many do not. Do not assume data follows a particular pattern just because a formula is familiar. Plot it and inspect it first.

### Practical activity

Make histograms for sales amount and order quantity. Compare their shapes. Which has a long tail? Which values might deserve a closer look?

## 6. Use a sample to estimate a wider group

A sample result is an estimate, so it has uncertainty. If we take a different sample, the result will usually change a little.

The **standard error** describes how much a sample estimate would typically vary across repeated samples. Larger, well-collected samples usually give more stable estimates. The **Central Limit Theorem** is the reason sample averages often form a bell-shaped pattern when we repeat sampling many times. We will use this idea, not prove it.

### Confidence intervals

A confidence interval gives a range of reasonable values for a population estimate. A 95% confidence method would capture the true value in about 95 out of 100 repeated studies, if the method’s assumptions are met.

```python
from scipy import stats

ratings = pd.Series([7, 8, 6, 9, 8, 7, 10, 6, 8, 9])
mean = ratings.mean()
se = stats.sem(ratings)
low, high = stats.t.interval(0.95, len(ratings) - 1, loc=mean, scale=se)
print(f"Average rating: {mean:.1f}")
print(f"95% confidence interval: {low:.1f} to {high:.1f}")
```

Say what the interval measures and include the units. Avoid saying that 95% of individual ratings fall inside the interval; it is about the estimated average.

### Practical exercise

Survey a group of customers about satisfaction. Report the sample size, average rating, and confidence interval. Would the interval be narrower or wider with more responses? Why?

## 7. Compare two groups carefully

A difference in sample averages may be due to a real difference, random variation, or both. First look at the group sizes, averages, and spread. Then use a statistical test if the question calls for one.

### A/B test example

Suppose a shop tests two page designs. Each visitor sees one design, and we record whether they sign up. Compare the signup proportions:

```python
results = pd.DataFrame({
    "page": ["A"] * 10 + ["B"] * 10,
    "signed_up": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                  1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
})
print(results.groupby("page")["signed_up"].agg(["count", "mean"]))
```

The mean of a yes/no column is the share of “yes” results. Before deciding, ask whether visitors were assigned fairly and whether the difference is useful in practice.

### Hypothesis test and p-value

A hypothesis test asks whether the data would be surprising if there were no real group difference.

- **Null hypothesis:** there is no difference in the wider groups.
- **Alternative hypothesis:** there is a difference.
- **P-value:** how surprising results this large or larger would be if the null hypothesis were true.

A small p-value is evidence against the no-difference explanation. It does not tell us the chance that the null hypothesis is true, and it does not tell us whether the difference is important to the business.

For a numeric outcome, a t-test is one common way to compare two group averages. For a yes/no outcome, compare proportions with a suitable proportion test. Use software for the calculation, then explain the result in everyday language. In this introductory course, we focus on choosing the question and interpreting the output rather than deriving test formulas.

### Practical checklist

- What was measured, and in which groups?
- How large is the observed difference?
- How many observations are in each group?
- Was the data collected fairly?
- Is the difference useful, even if it is statistically detectable?

## 8. Explore relationships and make simple predictions

A **scatter plot** shows how two numeric variables move together. Correlation summarizes how strongly a straight-line relationship appears. Correlation does not prove that one variable caused the other; another factor may affect both.

```python
plt.scatter(df["Quantity"], df["Sales"])
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.title("Quantity and sales")
plt.show()
print(df[["Quantity", "Sales"]].corr())
```

### Simple linear regression

Linear regression draws a line that summarizes the relationship between an input and a numeric outcome. For example, we might use quantity to estimate sales amount.

```python
from scipy.stats import linregress

clean = df[["Quantity", "Sales"]].dropna()
fit = linregress(clean["Quantity"], clean["Sales"])
print(f"Estimated sales change per extra item: {fit.slope:.2f}")
```

The slope describes the average change in the outcome for one more unit of the input in this dataset. It is not automatically a causal effect. Check the scatter plot and think about other factors before making a claim.

### Logistic regression: a brief introduction

When the outcome is yes/no, logistic regression estimates the chance of “yes.” Examples include whether a customer signs up or whether an email is spam. Treat this as a preview: the key idea is that the model outputs a probability between 0 and 1. Detailed model fitting can come later in machine learning.

### Practical exercise

Choose two columns from a dataset. Make a scatter plot or group comparison. Describe the pattern, then state one reason it may not show cause and effect.

## 9. Mini-project: answer one useful question

Work in pairs or small groups. Use the sales dataset or another familiar dataset.

1. Write one question that can be answered with the available columns.
2. Explain what one row represents and who is included in the data.
3. Check missing values and repeated rows.
4. Make one table and one chart that help answer the question.
5. Summarize a typical value or compare two groups.
6. If useful, calculate a confidence interval or run a simple test.
7. Present the result in three sentences: what you found, how certain you are, and one limitation.

A good conclusion is careful. For example: “In this sample, page B had a higher signup rate. The difference may be due to chance, and we need more fairly collected data before changing the site.”

## Quick reference

| Question | Useful starting point |
|---|---|
| What is typical? | Mean and median |
| How much does it vary? | IQR, standard deviation, histogram |
| How common is a category? | Count and proportion |
| How precise is an estimate? | Confidence interval |
| Do two groups look different? | Compare summaries, then consider a suitable test |
| Do two numeric columns move together? | Scatter plot and correlation |
| Can we estimate a numeric outcome? | Simple linear regression |
| Is the outcome yes/no? | Compare proportions; logistic regression is a later preview |

## Final reminders

- Start with a clear question.
- Look at the data before choosing a calculation.
- Explain results in plain words and include units.
- A relationship does not prove cause and effect.
- Share one limitation along with every conclusion.
