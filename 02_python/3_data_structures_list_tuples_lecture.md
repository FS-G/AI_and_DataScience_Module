**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Python Data Structures: Lists and Tuples
*From Single Items to Powerful Collections*

## Overview

This lecture introduces fundamental data structures in Python, covering:
- **Introduction to Data Structures**: Understanding why collections are needed over individual variables
- **Lists**: Mutable, ordered collections for storing and manipulating changeable data
- **Tuples**: Immutable, ordered collections for storing fixed data that shouldn't change
- **List Operations**: Creating, accessing, modifying, and iterating through lists using indexing, slicing, and methods
- **Tuple Operations**: Creating, accessing, and unpacking tuples, understanding immutability
- **List Comprehension**: Elegant Pythonic shortcuts for creating and transforming lists efficiently
- **Lists vs Tuples**: When to use each data structure based on mutability requirements
- **Practical Applications**: Real-world examples including grade management, data processing, and text manipulation

---

## Introduction to Data Structures

### Quick Recap: Control Flow and Functions

Remember how we made our programs smart with control flow? We learned to make decisions with `if/elif/else`, repeat actions with loops, organize code with functions, and handle errors gracefully. Now we're ready for the next level!

### The Need for Collections: Why Variables Aren't Enough

Imagine you're a teacher trying to store student names:

```python
# The painful way - individual variables
student1 = "Alice"
student2 = "Bob"
student3 = "Charlie"
student4 = "Diana"
student5 = "Eve"

# What if you have 100 students? 😱
# How do you loop through them?
# How do you add a new student?
```

**The problem:** Individual variables don't scale! We need a way to group related data together.

### Introduction to Lists and Tuples: Ordered Sequences

**Think of data structures like containers:**
- **Lists** 📝: Like a shopping list you can edit (add, remove, change items)
- **Tuples** 🔒: Like a birth certificate that never changes (fixed information)

Both are **ordered sequences** - they remember the order you put things in!

```python
# Much better approach!
students = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
coordinates = (10, 20)  # x, y position that shouldn't change
```

---

## The List: A Mutable Collection

### What is a List?

A **list** is like a smart container that:
- **Ordered**: Items have a specific position (1st, 2nd, 3rd...)
- **Mutable**: You can change, add, or remove items
- **Allows duplicates**: The same item can appear multiple times

### Creating a List

**Basic syntax:**
```python
my_list = [item1, item2, item3]
```

**Fun examples:**
```python
# Empty list
empty_list = []

# List of favorite foods
foods = ["pizza", "burger", "ice cream", "sushi"]

# List of numbers
scores = [85, 92, 78, 96, 88]

# Mixed types (Python allows this!)
mixed = ["Alice", 25, True, 3.14]

# List of lists (nested!)
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### Accessing Elements: Indexing

**Zero-based indexing** - Python starts counting from 0!

```python
fruits = ["apple", "banana", "cherry", "date"]

print(fruits[0])   # "apple" (first item)
print(fruits[1])   # "banana" (second item)
print(fruits[2])   # "cherry" (third item)
print(fruits[3])   # "date" (fourth item)
```

**Negative indexing** - Count from the end!

```python
fruits = ["apple", "banana", "cherry", "date"]

print(fruits[-1])  # "date" (last item)
print(fruits[-2])  # "cherry" (second to last)
print(fruits[-3])  # "banana" (third to last)
print(fruits[-4])  # "apple" (fourth to last)
```

**Memory trick:** Think of negative indexing as "from the back"!

### Slicing: Getting Sub-sections

**Syntax:** `my_list[start:stop:step]`

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic slicing
print(numbers[2:5])    # [2, 3, 4] (from index 2 to 4)
print(numbers[:4])     # [0, 1, 2, 3] (from start to index 3)
print(numbers[6:])     # [6, 7, 8, 9] (from index 6 to end)

# Step slicing
print(numbers[::2])    # [0, 2, 4, 6, 8] (every 2nd element)
print(numbers[1::2])   # [1, 3, 5, 7, 9] (every 2nd, starting from index 1)

# Reverse a list!
print(numbers[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

### Modifying a List: Mutability in Action

**Changing elements:**
```python
colors = ["red", "blue", "green"]
colors[1] = "yellow"  # Change "blue" to "yellow"
print(colors)  # ["red", "yellow", "green"]
```

**Adding elements:**
```python
shopping_list = ["milk", "eggs"]

# Add to the end
shopping_list.append("bread")
print(shopping_list)  # ["milk", "eggs", "bread"]

# Insert at specific position
shopping_list.insert(1, "butter")  # Insert at index 1
print(shopping_list)  # ["milk", "butter", "eggs", "bread"]
```

**Removing elements:**
```python
fruits = ["apple", "banana", "cherry", "banana"]

# Remove by value (removes first occurrence)
fruits.remove("banana")
print(fruits)  # ["apple", "cherry", "banana"]

# Remove by index (and get the removed item)
removed_item = fruits.pop(1)  # Remove item at index 1
print(f"Removed: {removed_item}")  # "cherry"
print(fruits)  # ["apple", "banana"]

# Remove last item
last_item = fruits.pop()
print(f"Last item: {last_item}")  # "banana"
print(fruits)  # ["apple"]
```

### Common List Methods

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# Get length
print(len(numbers))  # 8

# Count occurrences
print(numbers.count(1))  # 2 (how many times 1 appears)

# Sort the list (modifies original)
numbers.sort()
print(numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]

# Reverse the list
numbers.reverse()
print(numbers)  # [9, 6, 5, 4, 3, 2, 1, 1]

# Find index of item
print(numbers.index(5))  # 2 (index where 5 is located)
```

### Looping Through a List

```python
fruits = ["apple", "banana", "cherry"]

# Method 1: Simple loop
for fruit in fruits:
    print(f"I love {fruit}! 🍎")

# Method 2: Loop with index
for i, fruit in enumerate(fruits):
    print(f"{i+1}. {fruit}")

# Method 3: Using range and len
for i in range(len(fruits)):
    print(f"Fruit at index {i}: {fruits[i]}")
```

**Fun example: Grade Calculator**
```python
def calculate_average(grades):
    total = 0
    for grade in grades:
        total += grade
    return total / len(grades)

student_grades = [85, 92, 78, 96, 88]
average = calculate_average(student_grades)
print(f"Average grade: {average:.1f}")  # 87.8
```

---

## The Tuple: An Immutable Collection

### What is a Tuple?

A **tuple** is like a sealed container that:
- **Ordered**: Items have a specific position
- **Immutable**: Once created, you CANNOT change it
- **Allows duplicates**: Same item can appear multiple times

**Think of it as:** A permanent record that shouldn't be altered!

### Creating a Tuple

**Basic syntax:**
```python
my_tuple = (item1, item2, item3)
```

**Examples:**
```python
# Empty tuple
empty_tuple = ()

# Coordinates (x, y)
point = (10, 20)

# RGB color values
red_color = (255, 0, 0)

# Person's basic info
person = ("Alice", 25, "Engineer")

# Special case: Single-element tuple (note the comma!)
single_tuple = (42,)  # Without comma, it's just parentheses!
```

### Accessing Elements: Same as Lists

```python
colors = ("red", "green", "blue")

# Indexing
print(colors[0])   # "red"
print(colors[-1])  # "blue"

# Slicing
print(colors[1:])  # ("green", "blue")
```

### Immutability: The Key Difference

**This is what makes tuples special:**

```python
point = (10, 20)

# This will cause a TypeError!
try:
    point[0] = 15  # ❌ Cannot change tuple elements
except TypeError as e:
    print(f"Error: {e}")

# But you can create a new tuple
new_point = (15, 20)
print(new_point)  # ✅ This works fine
```

**Why is immutability useful?**
- **Data integrity**: Ensures important data doesn't accidentally change
- **Safety**: Multiple parts of your program can use the same tuple without worry
- **Dictionary keys**: Tuples can be used as dictionary keys (lists cannot!)

### Tuple Methods: Fewer but Useful

```python
numbers = (1, 2, 3, 2, 4, 2, 5)

# Count occurrences
print(numbers.count(2))  # 3

# Find index of first occurrence
print(numbers.index(4))  # 4

# Length still works
print(len(numbers))      # 7
```

### Tuple Unpacking: A Powerful Pythonic Feature

**Unpacking** means extracting values from a tuple into separate variables:

```python
# Basic unpacking
point = (10, 20)
x, y = point  # x gets 10, y gets 20
print(f"X: {x}, Y: {y}")

# RGB color unpacking
color = (255, 128, 0)
red, green, blue = color
print(f"Red: {red}, Green: {green}, Blue: {blue}")

# Swapping variables (Python magic!)
a = 5
b = 10
a, b = b, a  # Swap values!
print(f"a: {a}, b: {b}")  # a: 10, b: 5
```

**Function returns multiple values:**
```python
def get_name_age():
    return "Alice", 25  # Returns a tuple

name, age = get_name_age()  # Unpack the tuple
print(f"Name: {name}, Age: {age}")
```

---

## List Comprehension: Python's Elegant Shortcut

### What is List Comprehension?

**List comprehension** is Python's way of creating lists using a concise, readable syntax. It's like writing a mini-program inside square brackets!

**Think of it as:** A one-liner that does what a for loop does, but more elegantly.

### Basic Syntax

**Traditional way (for loop):**
```python
# Create a list of squares
squares = []
for i in range(5):
    squares.append(i ** 2)
print(squares)  # [0, 1, 4, 9, 16]
```

**List comprehension way:**
```python
# Same result, but much cleaner!
squares = [i ** 2 for i in range(5)]
print(squares)  # [0, 1, 4, 9, 16]
```

**Syntax breakdown:** `[expression for item in iterable]`

### Simple Examples

```python
# Create a list of numbers 0-9
numbers = [i for i in range(10)]
print(numbers)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create a list of even numbers
evens = [i for i in range(10) if i % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# Create a list of strings
fruits = ["apple", "banana", "cherry"]
upper_fruits = [fruit.upper() for fruit in fruits]
print(upper_fruits)  # ['APPLE', 'BANANA', 'CHERRY']

# Create a list of lengths
word_lengths = [len(word) for word in fruits]
print(word_lengths)  # [5, 6, 6]
```

### Conditional List Comprehension

**Adding conditions with `if`:**

```python
# Only even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [num for num in numbers if num % 2 == 0]
print(evens)  # [2, 4, 6, 8, 10]

# Only positive numbers
mixed_numbers = [-3, 1, -5, 8, -2, 9]
positives = [num for num in mixed_numbers if num > 0]
print(positives)  # [1, 8, 9]

# Only long words (more than 4 characters)
words = ["cat", "dog", "elephant", "bird", "giraffe"]
long_words = [word for word in words if len(word) > 4]
print(long_words)  # ['elephant', 'giraffe']
```

### Advanced Examples

**Nested list comprehension:**
```python
# Create a 3x3 matrix
matrix = [[i + j for j in range(3)] for i in range(0, 9, 3)]
print(matrix)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Flatten a nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```



### Real-World Applications

**Data processing:**
```python
# Process grades - only passing grades (>= 60)
grades = [85, 92, 45, 78, 96, 55, 88]
passing_grades = [grade for grade in grades if grade >= 60]
print(f"Passing grades: {passing_grades}")  # [85, 92, 78, 96, 88]

# Calculate letter grades
def get_letter_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

letter_grades = [get_letter_grade(grade) for grade in grades]
print(f"Letter grades: {letter_grades}")  # ['B', 'A', 'F', 'C', 'A', 'F', 'B']
```

**Text processing:**
```python
# Extract words that start with a vowel
sentence = "The quick brown fox jumps over the lazy dog"
words = sentence.split()
vowel_words = [word for word in words if word[0].lower() in 'aeiou']
print(vowel_words)  # ['over']

# Capitalize first letter of each word
capitalized = [word.capitalize() for word in words]
print(capitalized)  # ['The', 'Quick', 'Brown', 'Fox', 'Jumps', 'Over', 'The', 'Lazy', 'Dog']
```

### Performance Benefits

**List comprehension vs traditional loops:**

```python
import time

# Traditional way
start_time = time.time()
squares = []
for i in range(1000000):
    squares.append(i ** 2)
traditional_time = time.time() - start_time

# List comprehension way
start_time = time.time()
squares = [i ** 2 for i in range(1000000)]
comprehension_time = time.time() - start_time

print(f"Traditional loop: {traditional_time:.4f} seconds")
print(f"List comprehension: {comprehension_time:.4f} seconds")
# List comprehension is typically faster!
```

### When to Use List Comprehension

**✅ Use list comprehension when:**
- Creating a new list from an existing iterable
- Applying a simple transformation to each element
- Filtering elements based on a condition
- The logic is straightforward and readable

**❌ Avoid list comprehension when:**
- The logic is complex or hard to read
- You need side effects (like printing)
- The expression becomes too long (more than 80 characters)
- You need to use multiple statements

### Common Patterns

```python
# Pattern 1: Transform each element
numbers = [1, 2, 3, 4, 5]
doubled = [num * 2 for num in numbers]

# Pattern 2: Filter elements
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [num for num in numbers if num % 2 == 0]

# Pattern 3: Transform and filter
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_squares = [num ** 2 for num in numbers if num % 2 == 0]


```

### Practice Examples

**🎯 Challenge 1: Create a list of perfect squares up to 100**
```python
perfect_squares = [i ** 2 for i in range(1, 11)]  # 1, 4, 9, 16, 25, 36, 49, 64, 81, 100
```

**🎯 Challenge 2: Find all words in a sentence that are longer than 3 characters**
```python
sentence = "Python is a great programming language"
long_words = [word for word in sentence.split() if len(word) > 3]
# ['Python', 'great', 'programming', 'language']
```

**🎯 Challenge 3: Create a list of student names with their grades, but only for students who passed**
```python
students = [("Alice", 85), ("Bob", 45), ("Charlie", 92), ("Diana", 55)]
passing_students = [(name, grade) for name, grade in students if grade >= 60]
# [('Alice', 85), ('Charlie', 92)]
```

---

## Lists vs. Tuples - When to Use Which?

### A Practical Comparison

| Feature | List | Tuple |
|---------|------|-------|
| **Mutability** | ✅ Can change | ❌ Cannot change |
| **Syntax** | `[1, 2, 3]` | `(1, 2, 3)` |
| **Performance** | Slower | Faster |
| **Memory** | More memory | Less memory |
| **Methods** | Many methods | Few methods |
| **Use as dict key** | ❌ No | ✅ Yes |

### Use a List when:

**You have a collection that needs to change over time:**

```python
# To-do list (add/remove tasks)
todo_list = ["buy milk", "walk dog", "study Python"]
todo_list.append("call mom")  # ✅ List is perfect here

# Player scores in a game (scores change)
player_scores = [100, 250, 180]
player_scores[0] = 150  # ✅ Update player 1's score

# Shopping cart (add/remove items)
cart = ["laptop", "mouse"]
cart.remove("mouse")  # ✅ Changed mind about mouse
```

### Use a Tuple when:

**You have a collection that should NOT change:**

```python
# Coordinates (x, y position shouldn't change unexpectedly)
player_position = (100, 200)

# RGB color values (color definition is fixed)
brand_color = (255, 0, 0)  # Red

# Person's date of birth (never changes)
birth_date = (1995, 5, 15)  # Year, month, day

# Database configuration (should be stable)
db_config = ("localhost", 5432, "mydb")  # host, port, database
```

**Real-world example: Student Management System**
```python
# Student info that shouldn't change
student_info = ("John Doe", 12345, "Computer Science")  # name, ID, major

# Student grades that can change
student_grades = [85, 92, 78]  # can add new grades
student_grades.append(95)  # ✅ Added new grade

# Cannot accidentally change student info
# student_info[0] = "Jane Doe"  # ❌ This would cause an error
```

---

## Common Error Types: Quick Reference

```python
# IndexError - Accessing non-existent index
my_list = [1, 2, 3]
print(my_list[5])  # Index 5 doesn't exist

# TypeError - Trying to modify immutable tuple
my_tuple = (1, 2, 3)
my_tuple[0] = 5  # Cannot modify tuple

# ValueError - Item not in list
colors = ["red", "blue", "green"]
colors.remove("yellow")  # "yellow" not in list

# AttributeError - Wrong method for data type
my_tuple = (1, 2, 3)
my_tuple.append(4)  # Tuples don't have append method
```

---

## Wrap-up

### Summary of Topics Covered

**🎯 Today's Journey:**

1. **Data Structures Introduction** - Why we need collections over individual variables
2. **Lists** - Mutable, ordered collections perfect for changing data
3. **Tuples** - Immutable, ordered collections perfect for fixed data
4. **List Comprehension** - Elegant shortcuts for creating and transforming lists
5. **Key Operations** - Creating, accessing, modifying, and iterating through collections
6. **Practical Usage** - When to choose lists vs tuples

**🚀 Key Takeaways:**
- **Lists** are your go-to for data that changes (shopping lists, scores, tasks)
- **Tuples** are perfect for data that shouldn't change (coordinates, colors, IDs)
- **List comprehension** provides elegant shortcuts for creating and transforming lists
- **Indexing** starts from 0, negative indexing counts from the end
- **Slicing** lets you extract portions of sequences
- **Tuple unpacking** is a powerful Python feature for cleaner code

**🎮 Final Challenge:**
Create a simple grade management system that:
- Uses a tuple to store student information (name, ID, major)
- Uses a list to store and manage grades
- Calculates and displays the average grade
- Handles adding new grades and displaying all grades

**💡 Solution:**
```python
def grade_management_system():
    # Student info (shouldn't change) - use tuple
    student_info = ("Alice Johnson", 12345, "Computer Science")
    
    # Grades (can change) - use list
    grades = [85, 92, 78, 96]
    
    print(f"Student: {student_info[0]}")
    print(f"ID: {student_info[1]}")
    print(f"Major: {student_info[2]}")
    print("-" * 30)
    
    while True:
        print("\n1. View grades")
        print("2. Add grade")
        print("3. Calculate average")
        print("4. Exit")
        
        try:
            choice = input("Choose an option: ")
            
            if choice == "1":
                print(f"Grades: {grades}")
            elif choice == "2":
                new_grade = float(input("Enter new grade: "))
                grades.append(new_grade)
                print(f"Added grade: {new_grade}")
            elif choice == "3":
                if grades:
                    average = sum(grades) / len(grades)
                    print(f"Average grade: {average:.1f}")
                else:
                    print("No grades available!")
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid option!")
                
        except ValueError:
            print("Please enter a valid number!")

# Run the system
grade_management_system()
```

**What this solution demonstrates:**
- ✅ **Tuple**: For fixed student information
- ✅ **List**: For changeable grades
- ✅ **List methods**: append() for adding grades
- ✅ **Built-in functions**: len(), sum() for calculations
- ✅ **Error handling**: try/except for invalid input
- ✅ **Loops**: while loop for menu system

**Remember:** Choose your data structure based on whether your data needs to change! 📊✨

---

### Quick Reference Card

```python
# Lists (Mutable)
my_list = [1, 2, 3]
my_list[0] = 10        # Change element
my_list.append(4)      # Add to end
my_list.insert(1, 5)   # Insert at index
my_list.remove(2)      # Remove by value
item = my_list.pop()   # Remove by index

# Tuples (Immutable)
my_tuple = (1, 2, 3)
x, y, z = my_tuple     # Unpack
my_tuple.count(1)      # Count occurrences
my_tuple.index(2)      # Find index

# Common operations
len(sequence)          # Get length
for item in sequence:  # Loop through items
sequence[start:stop]   # Slice

# List Comprehension
[x for x in range(5)]           # [0, 1, 2, 3, 4]
[x for x in range(10) if x % 2 == 0]  # Even numbers
[x.upper() for x in ['a', 'b', 'c']]  # Transform elements
```

*Happy coding with collections! 🐍📦*

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*