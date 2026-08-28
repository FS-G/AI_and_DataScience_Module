**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Excel Lookup Functions Guide

## What Are Lookup Functions?

Lookup functions in Excel help you find and retrieve data from tables or ranges. Think of them as Excel's way of asking "find this information for me" - like looking up a phone number in a contact list or finding a product price in a catalog.

## The Two Main Lookup Functions

### 1. VLOOKUP (Vertical Lookup)
**Purpose:** Searches for a value in the first column of a table and returns a value from another column in the same row.

**Syntax:** `=VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])`

**Example Table - Employee Database:**
| Employee ID | Name        | Department | Salary  |
|-------------|-------------|------------|---------|
| 101         | John Smith  | Sales      | $50,000 |
| 102         | Jane Doe    | Marketing  | $55,000 |
| 103         | Bob Johnson | IT         | $60,000 |
| 104         | Alice Brown | HR         | $52,000 |

**Formula Example:** `=VLOOKUP(102, A2:D5, 2, FALSE)`
- Looks for Employee ID 102
- Searches in range A2:D5
- Returns value from column 2 (Name)
- Result: "Jane Doe"

### 2. HLOOKUP (Horizontal Lookup)
**Purpose:** Searches for a value in the first row of a table and returns a value from another row in the same column.

**Syntax:** `=HLOOKUP(lookup_value, table_array, row_index_num, [range_lookup])`

**Example Table - Monthly Sales:**
| Month   | Jan     | Feb     | Mar     | Apr     |
|---------|---------|---------|---------|---------|
| Sales   | $10,000 | $12,000 | $15,000 | $13,000 |
| Expenses| $8,000  | $9,000  | $11,000 | $10,000 |

**Formula Example:** `=HLOOKUP("Mar", A1:E2, 2, FALSE)`
- Looks for "Mar" in the first row
- Returns value from row 2 (Sales)
- Result: $15,000

## Parameter Breakdown

### VLOOKUP/HLOOKUP Parameters:
- **lookup_value:** What you're searching for
- **table_array:** The range containing your data
- **col_index_num/row_index_num:** Which column/row to return data from
- **range_lookup:** TRUE (approximate match) or FALSE (exact match)

## Common Use Cases with Examples

### Price Lookup System
**Product Table:**
| Product Code | Product Name | Price |
|--------------|--------------|-------|
| A001         | Laptop       | $899  |
| A002         | Mouse        | $25   |
| A003         | Keyboard     | $75   |

**Formula:** `=VLOOKUP("A002", A2:C4, 3, FALSE)`
**Result:** $25

### Grade Lookup System
**Student Scores:**
| Student | Math | Science | English |
|---------|------|---------|---------|
| Alice   | 85   | 92      | 78      |
| Bob     | 78   | 85      | 88      |
| Carol   | 92   | 78      | 85      |

**Formula:** `=VLOOKUP("Bob", A2:D4, 2, FALSE)`
**Result:** 78 (Bob's Math score)

## Pro Tips for Success

### Always Use FALSE for Exact Matches
Unless you specifically need approximate matching, always use FALSE as the last parameter in VLOOKUP/HLOOKUP.

### Handle Errors Gracefully
Wrap your lookup in IFERROR to handle missing values:
`=IFERROR(VLOOKUP(A2, DataTable, 2, FALSE), "Not Found")`

### Common Mistakes to Avoid
- Forgetting to use absolute references ($) when copying formulas
- Not matching data types (text vs numbers)
- Including headers in your lookup range
- Using approximate match when you need exact match

## Quick Reference Comparison

| Function | Best For | Limitation |
|----------|----------|------------|
| VLOOKUP | Vertical data, lookup column is leftmost | Can't look left |
| HLOOKUP | Horizontal data, lookup row is topmost | Can't look up |

## Practice Exercise

Create a simple inventory lookup using this data:

| Item ID | Item Name | Stock | Price |
|---------|-----------|-------|-------|
| SKU001  | Pen       | 150   | $2    |
| SKU002  | Notebook  | 75    | $8    |
| SKU003  | Stapler   | 25    | $15   |

Try these formulas:
1. Find the stock level for SKU002: `=VLOOKUP("SKU002", A2:D4, 3, FALSE)`
2. Find the price for Notebook: `=VLOOKUP("Notebook", B2:D4, 3, FALSE)`
3. Create an error-handled lookup for a non-existent item: `=IFERROR(VLOOKUP("SKU999", A2:D4, 2, FALSE), "Item Not Found")`

## Additional Tips for Better Lookups

### Data Validation with Lookups
Use lookups to create dynamic dropdown lists and data validation rules that automatically update when your source data changes.

### Performance Considerations
For large datasets, consider using exact match (FALSE) and ensure your lookup table is sorted if using approximate match (TRUE) for better performance.

Mastering these lookup functions will dramatically improve your Excel efficiency and make you much more productive with data analysis tasks.

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*