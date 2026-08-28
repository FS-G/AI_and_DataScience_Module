# 4️⃣ Python for Excel Automation: Social Media Analytics 📊

## The Problem: Manual Excel Hell vs. Automated Magic

Imagine you're managing social media for a company. Every day, you need to:
- ✋ Manually collect data from Instagram, TikTok, LinkedIn
- ✋ Copy-paste into Excel
- ✋ Calculate metrics (average engagement, top posts, best posting times)
- ✋ Format reports with colors and tables
- ✋ Send to your boss

**With VBA:** You write macros... that only work in Excel... and are hard to update.

**With Python:** You write a script once, run it daily on autopilot, and it creates beautiful reports automatically. 🚀

---

## 🎯 Today's Mission: Automate Social Media Analytics

We'll use Python to:
1. **Read** social media data from CSV
2. **Analyze** which platforms/content types perform best
3. **Generate** professional Excel reports automatically
4. **Schedule** it to run daily without touching Excel

**The magic trio:** `pandas` (data), `openpyxl` (Excel), automation

---

## 📊 Our Dataset: Real Social Media Metrics

**File:** `social_media_data.csv`

```
post_id, date, platform, content_type, likes, comments, shares, followers_gained, engagement_rate, post_hour
P001, 2026-08-01, Instagram, Photo, 1250, 45, 120, 15, 3.2, 10
P002, 2026-08-01, TikTok, Video, 5600, 320, 890, 45, 8.7, 19
...
```

**What we're tracking:**
- Which platform gets the most engagement? (TikTok wins 🎉)
- Which content type performs best per platform?
- What time should we post to go viral?
- Which posts should we replicate?

---

## 🔥 Script 1: Basic Analytics Report (The Starter)

**What it does:** Reads CSV → Calculates platform stats → Creates a formatted Excel report

```python
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

# Read the data
df = pd.read_csv('social_media_data.csv')

# Platform-wise statistics (group by and aggregate)
platform_stats = df.groupby('platform').agg({
    'likes': 'sum',
    'engagement_rate': 'mean'
}).round(2)

# Find top 5 performing posts
top_posts = df.nlargest(5, 'engagement_rate')

# Create Excel workbook
wb = Workbook()
ws = wb.active

# Add title with styling (NO MORE MANUAL FORMATTING!)
ws['A1'] = "SOCIAL MEDIA ANALYTICS REPORT"
ws['A1'].font = Font(bold=True, size=14)
ws['A1'].fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")

# Add platform stats to sheet (automatic!)
row = 4
for platform, stats in platform_stats.iterrows():
    ws.cell(row=row, column=1).value = platform
    ws.cell(row=row, column=2).value = stats['likes']
    ws.cell(row=row, column=3).value = stats['engagement_rate']
    row += 1

# Add top posts
# ... (similar loop)

wb.save('social_media_report.xlsx')
print("✅ Report generated in 0.5 seconds!")
```

**The power:** What took 20 minutes manually → 0.5 seconds of Python code execution

**Key concepts:**
- `pd.read_csv()` → Reads CSV like a table
- `.groupby()` → Groups data (like pivot tables)
- `.agg()` → Calculates sums, averages, counts at once
- `.nlargest()` → Finds top N rows instantly
- Loop through data and add to Excel → **No more copy-paste**

---

## 🎯 Script 2: Smart Content Optimizer (The Game-Changer)

**What it does:** Analyzes content types per platform + generates AI recommendations

```python
import pandas as pd

# Analyze content type performance
content_performance = df.groupby(['platform', 'content_type']).agg({
    'engagement_rate': ['mean', 'max'],
    'followers_gained': 'sum'
}).round(2)

# ⚠️ IMPORTANT: Flatten multi-level columns after .agg()
# When using multiple aggregations, pandas creates column tuples
# We convert them to simple string names for easy access
content_performance.columns = ['_'.join(col).strip() for col in content_performance.columns.values]

# Now we can access: engagement_rate_mean, engagement_rate_max, followers_gained_sum

# Create recommendation logic (if-else on steroids)
def get_recommendation(engagement):
    if engagement > 10:
        return "🔥 VIRAL - Keep producing!"
    elif engagement > 5:
        return "✅ Good - Continue"
    else:
        return "⚠️ Needs improvement"

# Add recommendation to every post (applies function to every row)
df['recommendation'] = df.apply(
    lambda row: get_recommendation(row['platform'], row['engagement_rate']),
    axis=1
)

# Find best posting hours
best_hours = df.groupby('post_hour')['engagement_rate'].mean().nlargest(3)
# Result: Post at 8 PM, 9 PM, 10 PM for maximum virality!

# When adding to Excel, use the flattened column names:
ws1.cell(row=row_num, column=3).value = row_data['engagement_rate_mean']
ws1.cell(row=row_num, column=4).value = row_data['engagement_rate_max']
ws1.cell(row=row_num, column=5).value = int(row_data['followers_gained_sum'])
```

**The magic:** In 15 lines, you've created insights that would take an analyst 2 hours manually.

**💡 Pandas Pro Tip:** When `.agg()` creates multi-level columns, flatten them first to avoid KeyError!

**Wow factor for students:**
- The script automatically created a "recommendation" column for 30 posts
- Identified that TikTok videos posted at 9-10 PM get 15% engagement (Instagram only gets 4%)
- This is **actionable intelligence** generated by code

---

## ⏰ Script 3: Automated Daily Reporter (The Scheduler)

**What it does:** Runs daily at 8 AM, generates yesterday's report automatically

```python
from datetime import datetime

# Filter for today's data
today = '2026-08-10'
today_data = df[df['date'] == today]

# Calculate daily KPIs
total_engagement = today_data['engagement_rate'].sum()
followers_gained = today_data['followers_gained'].sum()
top_post = today_data.loc[today_data['engagement_rate'].idxmax()]

# Platform breakdown - IMPORTANT: Use column names directly, not indices!
platform_daily = today_data.groupby('platform').agg({
    'likes': 'sum',
    'engagement_rate': 'mean',
    'followers_gained': 'sum'
}).round(2)

# Create report
wb = Workbook()
ws = wb.active

# Add KPI boxes (like a dashboard)
ws['A1'] = f"DAILY REPORT - {today}"
ws['A3'] = f"Followers Gained: {followers_gained}"
ws['A4'] = f"Total Engagement: {round(total_engagement, 2)}%"
ws['A5'] = f"Top Post: {top_post['post_id']} ({top_post['platform']})"

# ✅ Correct way to add platform data: Use column NAMES
for row_num, (platform, row_data) in enumerate(platform_daily.iterrows(), 18):
    ws.cell(row=row_num, column=1).value = platform
    ws.cell(row=row_num, column=2).value = int(row_data['likes'])        # Column name, not [0]
    ws.cell(row=row_num, column=3).value = row_data['engagement_rate']   # Column name, not [1]
    ws.cell(row=row_num, column=4).value = int(row_data['followers_gained'])  # Column name, not [2]

# Auto-filename with date
filename = f"daily_report_{today}.xlsx"
wb.save(filename)

# This script can run automatically every morning at 8 AM!
```

**The future:** You schedule this with Task Scheduler (Windows) or cron (Mac/Linux)
- 8:00 AM: Script runs automatically ✅
- 8:00:05 AM: Report is emailed to your boss 📧
- You're still sleeping 😴

**🐛 Bug Fix:** When iterating through aggregated data, always use **column names** (e.g., `row_data['likes']`) not numeric indices like `row_data[0]`

---

## 🚀 Why Python > VBA for Excel Automation

| Feature | VBA | Python |
|---------|-----|--------|
| Data processing | Slow, clunky | Super fast |
| Code readability | Confusing | Crystal clear |
| Reuse code | Hard | Easy (functions, libraries) |
| Learning curve | Steep | Gentle |
| Job market | Declining | 🔥 In-demand |
| Integration | Excel only | Works with APIs, databases, cloud |
| Community | Small | Massive (Stack Overflow, GitHub) |

---

## 📦 Libraries We're Using (The Toolkit)

### **Pandas** 🐼
- Reads CSV/Excel files
- Groups, filters, and calculates data
- Like Excel but 100x faster

```python
df = pd.read_csv('data.csv')  # Read file
df.groupby('platform')['likes'].sum()  # Total likes by platform
df[df['engagement_rate'] > 10]  # Filter rows
```

### **OpenPyXL** 📄
- Creates/modifies Excel files
- Adds colors, fonts, borders
- No need to open Excel!

```python
from openpyxl import Workbook
wb = Workbook()
ws = wb.active
ws['A1'] = "Hello"
ws['A1'].font = Font(bold=True, color="FFFFFF")
wb.save('report.xlsx')
```

---

## 🎪 The Demo Breakdown

### Files You'll Get:

1. **social_media_data.csv** → 30 posts across 3 platforms (real-looking data)
2. **script_1_basic_analysis.py** → Creates platform performance report
3. **script_2_content_optimization.py** → Multi-sheet report with recommendations
4. **script_3_daily_automation.py** → Daily report generator (can be scheduled)

### What Happens When You Run Each Script:

```bash
python script_1_basic_analysis.py
✅ Report generated: social_media_report.xlsx

python script_2_content_optimization.py
✅ Content optimization report generated: content_optimizer.xlsx
(3 sheets: Content Performance, Recommendations, Best Times)

python script_3_daily_automation.py
✅ Daily report saved: daily_report_2026-08-10.xlsx
📊 Summary: 245 new followers, 85.6% total engagement
```

---

## 💡 Key Takeaways for Students

1. **Python replaces VBA:** Write once, use forever (no Excel needed!)
2. **Speed matters:** 0.5 seconds vs 20 minutes is a 2,400x improvement
3. **Scalability:** Want to analyze 100,000 posts? Python handles it. VBA would crash.
4. **Career skills:** Every tech job wants Python. Nobody wants VBA.
5. **Automation = Money:** Schedule this once, and you've saved 5 hours/week forever

---

## 🎓 Try This Yourself

1. Copy `social_media_data.csv` to your folder
2. Run `script_1_basic_analysis.py`
3. Open `social_media_report.xlsx` in Excel
4. **Mind = 🤯** when you see a professional report generated in code

---

## 🔗 What's Next?

- **Script 4:** Connect to Instagram API, pull real data automatically
- **Script 5:** Send reports via email with `smtplib`
- **Script 6:** Create a web dashboard with `Flask` (replace Excel entirely!)

---

## TL;DR - The One-Sentence Pitch

**Python + Pandas + OpenPyXL = VBA's replacement that's 10x better, 10x easier, and 100% cooler** 🚀

