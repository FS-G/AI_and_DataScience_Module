import pandas as pd

# Load
q1 = pd.read_csv('sales_q1.csv')
q2 = pd.read_csv('sales_q2.csv')
customers = pd.read_csv('customers.csv')
costs = pd.read_csv('product_costs.csv')
targets = pd.read_csv('region_targets.csv')

# Merge
sales = pd.concat([q1, q2], ignore_index=True)
sales['revenue'] = sales['qty'] * sales['unit_price']

df = sales.merge(customers, on='order_id', how='left') \
          .merge(costs, on='product', how='left') \
          .merge(targets, on='region', how='left')

df['cost'] = df['qty'] * df['unit_cost']
df['profit'] = df['revenue'] - df['cost']

df.to_csv('merged_data.csv', index=False)

# Insights
print("=== INSIGHTS ===\n")

print("Total Revenue: PKR {:,.0f}".format(df['revenue'].sum()))
print("Total Profit: PKR {:,.0f}".format(df['profit'].sum()))
print("Total Orders:", len(df))

print("\n-- Revenue by Region --")
print(df.groupby('region')['revenue'].sum().sort_values(ascending=False).apply(lambda x: f"PKR {x:,.0f}"))

print("\n-- Revenue by Product --")
print(df.groupby('product')['revenue'].sum().sort_values(ascending=False).apply(lambda x: f"PKR {x:,.0f}"))

print("\n-- Profit Margin by Product --")
margin = df.groupby('product').apply(lambda g: (g['profit'].sum()/g['revenue'].sum())*100)
print(margin.round(1).astype(str) + '%')

print("\n-- Revenue by Customer Type --")
print(df.groupby('customer_type')['revenue'].sum().sort_values(ascending=False).apply(lambda x: f"PKR {x:,.0f}"))

print("\n-- Region vs Target (approx, single period vs monthly target) --")
region_rev = df.groupby('region')['revenue'].sum()
region_target = targets.set_index('region')['monthly_target_pkr']
comparison = pd.DataFrame({'revenue': region_rev, 'monthly_target': region_target})
comparison['pct_of_target'] = (comparison['revenue'] / comparison['monthly_target'] * 100).round(1)
print(comparison)

print("\n-- Top 5 Orders by Profit --")
print(df.nlargest(5, 'profit')[['order_id','region','product','profit']])