**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Python Introduction Lecture

## Overview

This lecture covers the fundamentals of Python programming, including:
- **Introduction to Python**: What Python is, why it's popular, and companies using it
- **Environment Setup**: Installing Python, setting up VS Code and Google Colab, virtual environments, and requirements.txt
- **First Programs**: Hello World, comments, and basic syntax
- **Variables and Data Types**: Working with strings, integers, floats, and booleans
- **Operators**: Arithmetic, comparison, and logical operators
- **User Input**: Getting input from users and type conversion
- **Practical Exercises**: Hands-on coding examples including calculators and data collection programs
- **Assignments**: Practice problems to reinforce learning

---

## Welcome and The "Why" of Python

### What is Python?
- High-level, interpreted, general-purpose programming language
- Created by Guido van Rossum in 1991
- Emphasis on code readability (the "Zen of Python")
- Named after "Monty Python's Flying Circus"

### Why is Python So Popular?
- **Versatility**: Web development, data science, AI/ML, automation, scripting
- **Beginner-friendly syntax**: Clean, readable code that resembles English
- **Massive community**: Extensive libraries and frameworks
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Open source**: Free to use and distribute

### Companies Using Python
- Google, Netflix, Instagram, Spotify
- NASA, Reddit, YouTube, Dropbox
- Scientific research institutions worldwide

---

## Setting Up Your Environment (Crucial First Step)

### Installation Options
- **Python.org**: Official Python installer
- **Anaconda**: Python distribution with data science packages
- **For this course**: We'll use Google Colab and VS Code

### Development Environments Overview
- **Google Colab**: Browser-based, no installation needed
- **VS Code**: Popular code editor with Python extension
- **PyCharm**: Full-featured IDE for Python
- **Jupyter Notebook**: Interactive computing environment
- **IDLE**: Comes built-in with Python

### VS Code Setup (Windows)
```bash
# Step 1: Install Python from python.org
# Step 2: Install VS Code
# Step 3: Install Python extension in VS Code

# Creating virtual environment
python -m venv myenv

# Activating virtual environment
# On Windows Command Prompt:
myenv\Scripts\activate

# On Windows PowerShell:
myenv\Scripts\Activate.ps1

# On Git Bash or Linux/Mac:
source myenv/Scripts/activate
```

### Common Permission Issues & Solutions
```bash
# If you get execution policy error in PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative: Use Command Prompt instead of PowerShell
# Or activate using Python directly:
python -m venv myenv
python myenv/Scripts/activate

# For Git Bash users:
source myenv/Scripts/activate
```

### Creating and Using requirements.txt
```bash
# Step 1: Create a requirements.txt file manually
# Example requirements.txt content:
requests
pandas
numpy
matplotlib

# Step 2: Install packages from requirements.txt
pip install -r requirements.txt

# Step 3: Verify installation
pip list
```

### Sample requirements.txt file
```
# Basic data science packages
pandas
numpy
matplotlib
requests
jupyter

# You can also specify versions if needed
# requests==2.28.1
# pandas==1.5.2
```

### Google Colab Setup
- Go to colab.research.google.com
- Sign in with Google account
- Create new notebook
- Ready to code!

---

## Your First Python Program - IN GOOGLE COLAB

### The "Hello, World!" Program
```python
# This is our first Python program
print("Hello, World!")
```

**What is a function?**
- A function is a command that does something
- `print()` is a function that displays text on screen
- Functions have parentheses `()` and may take inputs

### More print() Examples
```python
print("Welcome to Python!")
print("My name is Alice")
print(42)
print(3.14)
```

### Comments: Why They Are Important
- Help you remember what your code does
- Help others understand your code
- Make debugging easier

### Single-line Comments
```python
# This is a single-line comment
print("Hello")  # Comment at the end of a line

# Use comments to explain complex code
# or to temporarily disable code
# print("This won't run")
```

### Multi-line Comments / Docstrings
```python
"""
This is a multi-line comment.
It can span multiple lines.
Used for detailed explanations.
"""

'''
Another way to write
multi-line comments
using single quotes.
'''

def greet():
    """
    This is a docstring.
    It describes what the function does.
    """
    print("Hello from function!")
```

### Variables and Assignment

**What is a variable?**
- A named container for data
- Like a box with a label that holds something

```python
# Creating variables
name = "Alice"
age = 25
height = 5.6
is_student = True
```

**The assignment operator (=)**
- Used to assign values to variables
- Right side is evaluated first, then assigned to left side

```python
x = 10
y = 20
sum_result = x + y  # sum_result now contains 30
```

**Python's Dynamic Typing**
- No need to declare variable type
- Python figures out the type automatically

```python
# Python automatically determines types
message = "Hello"      # string
count = 42            # integer
price = 19.99         # float
is_valid = True       # boolean

# Variables can change type
x = 10        # x is integer
x = "hello"   # now x is string
```

**Variable Naming Conventions**
```python
# Good variable names (snake_case)
first_name = "John"
user_age = 30
total_price = 99.99

# Avoid these naming styles
firstName = "John"     # camelCase (less common in Python)
FirstName = "John"     # PascalCase (used for classes, not variables)
first-name = "John"    # Error: hyphens not allowed
2name = "John"         # Error: can't start with number
class = "Python"       # Error: can't use keywords
def = "definition"     # Error: can't use keywords
user name = "John"     # Error: spaces not allowed
user@name = "John"     # Error: special characters not allowed
my.variable = 10       # Error: dots not allowed
```

---

## Python's Fundamental Data Types

### Introduction to Data Types
- Data types are different kinds of information
- Python has several built-in data types
- Each type has different capabilities and uses

### Text: The String (str)

**Creating Strings**
```python
# Using single quotes
message = 'Hello, World!'
name = 'Alice'

# Using double quotes
greeting = "Welcome to Python"
quote = "She said, 'Hello there!'"

# Using triple quotes for multi-line
paragraph = """This is a long paragraph
that spans multiple lines.
Very useful for longer text."""
```

**Simple String Concatenation**
```python
first_name = "John"
last_name = "Doe"

# Using + operator
full_name = first_name + " " + last_name
print(full_name)  # Output: John Doe

# Using * for repetition
laugh = "Ha" * 3
print(laugh)  # Output: HaHaHa
```

**Introduction to f-strings (Easy Formatting)**
```python
name = "Alice"
age = 25
height = 5.6

# f-string formatting (recommended)
message = f"My name is {name} and I am {age} years old"
print(message)

# More f-string examples
price = 19.99
item = "book"
print(f"The {item} costs ${price}")
print(f"Next year, {name} will be {age + 1} years old")
```

### Numbers: Integers (int) and Floats (float)

**Integers (int) - Whole Numbers**
```python
age = 25
year = 2024
temperature = -5
big_number = 1000000

# Python can handle very large integers
huge_number = 12345678901234567890
print(huge_number)
```

**Floats (float) - Numbers with Decimal Points**
```python
price = 19.99
pi = 3.14159
temperature = -2.5
percentage = 0.85

# Scientific notation
speed_of_light = 3.0e8  # 3.0 x 10^8
print(speed_of_light)
```

**Working with Numbers**
```python
# Mixed operations
result = 10 + 3.5  # Result is 13.5 (float)
print(result)

# Converting between types
x = 10          # integer
y = float(x)    # convert to float: 10.0
z = int(3.7)    # convert to int: 3 (truncated)
```

### Truth Values: The Boolean (bool)

**Boolean Values**
```python
is_student = True
is_graduated = False
has_license = True

# Note: Capitalization matters!
# true and false (lowercase) are NOT valid
```

**Booleans from Comparisons**
```python
age = 18
is_adult = age >= 18  # True
print(is_adult)

# Other examples
print(5 > 3)        # True
print(10 == 10)     # True
print("a" == "A")   # False
```

### Checking the Type: Using the type() Function

**Inspecting Variable Types**
```python
name = "Alice"
age = 25
height = 5.6
is_student = True

# Check types
print(type(name))       # <class 'str'>
print(type(age))        # <class 'int'>
print(type(height))     # <class 'float'>
print(type(is_student)) # <class 'bool'>
```

**Practical Example**
```python
# Mixed data types
data = [42, "hello", 3.14, True]

for item in data:
    print(f"Value: {item}, Type: {type(item)}")
```

---

## Basic Operators and User Input

### Arithmetic Operators

**Basic Math Operations**
```python
a = 10
b = 3

print(a + b)    # Addition: 13
print(a - b)    # Subtraction: 7
print(a * b)    # Multiplication: 30
print(a / b)    # Division: 3.3333...
print(a ** b)   # Exponentiation: 1000
print(a // b)   # Floor division: 3
print(a % b)    # Modulo (remainder): 1
```

**Practical Examples**
```python
# Calculate area of rectangle
length = 10
width = 5
area = length * width
print(f"Area: {area}")

# Calculate compound interest
principal = 1000
rate = 0.05
time = 3
amount = principal * (1 + rate) ** time
print(f"Final amount: ${amount:.2f}")
```

### Comparison Operators

**All Comparison Operators**
```python
x = 10
y = 5

print(x == y)    # Equal to: False
print(x != y)    # Not equal to: True
print(x > y)     # Greater than: True
print(x < y)     # Less than: False
print(x >= y)    # Greater than or equal: True
print(x <= y)    # Less than or equal: False
```

**Comparing Different Types**
```python
# String comparisons
print("apple" == "banana")  # False
print("Apple" == "apple")   # False (case-sensitive)

# Mixed type comparisons (be careful!)
age_str = "25"
age_int = 25
print(age_str == str(age_int))  # True
print(age_str == age_int)       # False
```

### Logical Operators

**and, or, not Operators**
```python
age = 20
has_license = True
has_car = False

# AND operator (both must be True)
can_drive = age >= 18 and has_license
print(can_drive)  # True

# OR operator (at least one must be True)
can_travel = has_car or has_license
print(can_travel)  # True

# NOT operator (reverses the boolean)
is_minor = not (age >= 18)
print(is_minor)  # False
```

**Practical Example**
```python
# Admission criteria
gpa = 3.5
test_score = 85
extracurricular = True

# Complex condition
eligible = (gpa >= 3.0 and test_score >= 80) or extracurricular
print(f"Eligible for admission: {eligible}")
```

### Getting User Input

**The input() Function**
```python
# Getting string input
name = input("Enter your name: ")
print(f"Hello, {name}!")

# Getting multiple inputs
city = input("Enter your city: ")
country = input("Enter your country: ")
print(f"You are from {city}, {country}")
```

**Key Concept: input() Always Returns a String**
```python
# This is a string, not a number!
age_input = input("Enter your age: ")
print(type(age_input))  # <class 'str'>

# This will cause an error:
# next_year = age_input + 1  # Error: can't add string and int
```

**Type Casting: Converting Input**
```python
# Converting to integer
age_str = input("Enter your age: ")
age = int(age_str)
next_year = age + 1
print(f"Next year you'll be {next_year}")

# Converting to float
height_str = input("Enter your height in meters: ")
height = float(height_str)
height_cm = height * 100
print(f"Your height is {height_cm} cm")

# Direct conversion (more common)
weight = float(input("Enter your weight in kg: "))
bmi_height = float(input("Enter your height in meters: "))
bmi = weight / (bmi_height ** 2)
print(f"Your BMI is {bmi:.2f}")
```

**Handling Input Errors (Preview)**
```python
# What happens with invalid input?
try:
    number = int(input("Enter a number: "))
    print(f"You entered: {number}")
except ValueError:
    print("That's not a valid number!")
```

---

## Practical Exercises

### Exercise 1: Personal Information Collector
```python
# Collect user information
print("=== Personal Information Form ===")
first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")
age = int(input("Enter your age: "))
city = input("Enter your city: ")
is_student = input("Are you a student? (yes/no): ").lower() == "yes"

# Calculate birth year
current_year = 2024
birth_year = current_year - age

# Display information
print(f"\n=== Your Information ===")
print(f"Full Name: {first_name} {last_name}")
print(f"Age: {age} years old")
print(f"Birth Year: {birth_year}")
print(f"City: {city}")
print(f"Student Status: {is_student}")
```

### Exercise 2: Simple Calculator
```python
# Simple calculator
print("=== Simple Calculator ===")
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Perform all operations
addition = num1 + num2
subtraction = num1 - num2
multiplication = num1 * num2
division = num1 / num2 if num2 != 0 else "Cannot divide by zero"

# Display results
print(f"\n=== Results ===")
print(f"{num1} + {num2} = {addition}")
print(f"{num1} - {num2} = {subtraction}")
print(f"{num1} × {num2} = {multiplication}")
print(f"{num1} ÷ {num2} = {division}")
```

### Exercise 3: BMI Calculator
```python
# BMI Calculator
print("=== BMI Calculator ===")
name = input("Enter your name: ")
weight = float(input("Enter your weight in kg: "))
height = float(input("Enter your height in meters: "))

# Calculate BMI
bmi = weight / (height ** 2)

# Display result
print(f"\n=== BMI Results ===")
print(f"Name: {name}")
print(f"Weight: {weight} kg")
print(f"Height: {height} m")
print(f"BMI: {bmi:.2f}")

# Basic interpretation
if bmi < 18.5:
    category = "Underweight"
elif bmi < 25:
    category = "Normal weight"
elif bmi < 30:
    category = "Overweight"
else:
    category = "Obese"

print(f"Category: {category}")
```

---

## Summary and Next Steps

### What We Learned Today
- **Python basics**: What it is and why it's popular
- **Environment setup**: Google Colab and VS Code
- **First programs**: Hello World and comments
- **Variables**: Creating and naming conventions
- **Data types**: Strings, integers, floats, booleans
- **Operators**: Arithmetic, comparison, logical
- **User input**: Getting and converting input
- **Practical coding**: Real-world examples

### Key Concepts to Remember
- Variables are containers for data
- Python is dynamically typed
- `input()` always returns a string
- Use f-strings for easy formatting
- Comments make code readable
- Practice makes perfect!

### Practice Assignments

#### Assignment 1: Age Calculator
**Task**: Create a program that asks for someone's birth year and calculates their age.

```python
# Solution
birth_year = int(input("Enter your birth year: "))
current_year = 2024
age = current_year - birth_year
print(f"You are {age} years old")
```

#### Assignment 2: Name Formatter
**Task**: Ask for first name and last name, then display them in different formats.

```python
# Solution
first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")

print(f"Full name: {first_name} {last_name}")
print(f"Last, First: {last_name}, {first_name}")
print(f"Initials: {first_name[0]}.{last_name[0]}.")
```

#### Assignment 3: Circle Area Calculator
**Task**: Ask for radius and calculate the area of a circle.

```python
# Solution
radius = float(input("Enter the radius of circle: "))
pi = 3.14159
area = pi * radius ** 2
print(f"Area of circle: {area:.2f}")
```

#### Assignment 4: Temperature Converter
**Task**: Convert Celsius to Fahrenheit.

```python
# Solution
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit}°F")
```

### Resources for Continued Learning
- **Official Python Tutorial**: python.org/tutorial
- **Google Colab**: colab.research.google.com
- **VS Code Python Tutorial**: code.visualstudio.com/docs/python
- **Python.org Documentation**: docs.python.org

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*