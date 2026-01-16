**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Python Control Flow Lecture

## Overview

This lecture introduces control flow in Python, covering:
- **Introduction to Control Flow**: Understanding decision-making and repetition in programs
- **Conditional Statements**: Using `if`, `elif`, and `else` to make decisions based on conditions
- **Loops**: Implementing `for` and `while` loops for repetition, including `break` and `continue` statements
- **Functions**: Creating reusable code blocks with `def`, parameters, return values, and understanding variable scope
- **Error Handling**: Using `try...except` blocks to handle errors gracefully and prevent program crashes
- **Common Error Types**: Learning about ValueError, ZeroDivisionError, IndexError, and other common Python errors
- **Practical Examples**: Building interactive programs like guessing games, calculators, and validators

---

## Introduction to Control Flow

### What is Control Flow?

Imagine you're giving directions to a friend:
- "Go straight until you see a red light"
- "If the light is green, keep going"
- "If it's red, stop and wait"
- "Repeat this process until you reach the mall"

This is exactly what **control flow** does in programming! Instead of our program just running line by line like a boring grocery list, we can make it:
- Make decisions ("Should I do this or that?") → **Conditional statements**
- Repeat actions ("Do this 10 times") → **Loops**
- Handle problems ("What if something goes wrong?") → **Error handling**

**Control flow includes both conditional statements (if/elif/else) and loops (for/while) - they're the two main tools that make programs intelligent!**

**Think of it like this:** Your program becomes a smart assistant that can think, decide, and adapt!

---

## Conditional Statements

### The if Statement: Teaching Your Program to Think

Let's start with a fun example - a simple mood detector!

```python
mood = "happy"

if mood == "happy":
    print("😊 Great! Let's have a good day!")
```

**Key Points:**
- The `if` keyword starts our condition
- We use `==` to check if something equals something else
- The colon `:` is like saying "then do this:"

### The else Statement: Plan B

What if our mood isn't happy? We need a backup plan!

```python
mood = "sad"

if mood == "happy":
    print("😊 Great! Let's have a good day!")
else:
    print("😔 That's okay, tomorrow will be better!")
```

### The elif Statement: Multiple Choices

Life isn't just happy or sad - let's handle more moods!

```python
mood = "excited"

if mood == "happy":
    print("😊 Great! Let's have a good day!")
elif mood == "excited":
    print("🎉 Wow! You're full of energy!")
elif mood == "sleepy":
    print("😴 Maybe grab a coffee first?")
else:
    print("🤔 Whatever you're feeling, you've got this!")
```

### The Importance of Indentation: Python's Magic Rule

**CRITICAL CONCEPT:** Python uses spaces (indentation) to group code together!

```python
# ✅ CORRECT
if mood == "happy":
    print("This is inside the if block")
    print("This is also inside the if block")
print("This is outside the if block")

# ❌ WRONG
if mood == "happy":
print("This will cause an error!")
```



**Remember:** 4 spaces or 1 tab - be consistent!

### Live Coding Example: Age Classifier

Let's build something fun together!

```python
age = 16

if age < 13:
    print("You're a kid! 🧒")
elif age < 20:
    print("You're a teenager! 🧑‍🎓")
elif age < 60:
    print("You're an adult! 👩‍💼")
else:
    print("You're a wise senior! 👴")
```

### Example
make a program that takes in age and specify age brackets - in case if the participant is a kid (less than 7), offer him a toy. In case if the participant is a wise citizen (over 70), offer him a wheel chair 

```python
age = int(input("Enter Age")) # replace with input

if age < 13:
    print("You're a kid! 🧒")
    if age < 7:
        print("Do you want a toy?")
elif age < 20:
    print("You're a teenager! 🧑‍🎓")
elif age < 60:
    print("You're an adult! 👩‍💼")
else:
    print("You're a wise senior! 👴")
    if age > 70:
      print("Do you want a wheel chair?")
```



---

## Loops for Repetition

### Why We Need Loops: The DRY Principle

**DRY = Don't Repeat Yourself**

Instead of writing:
```python
print("Happy Birthday! 🎂")
print("Happy Birthday! 🎂")
print("Happy Birthday! 🎂")
print("Happy Birthday! 🎂")
print("Happy Birthday! 🎂")
```

We can write:
```python
for i in range(5):
    print("Happy Birthday! 🎂")
```

### The for Loop: Doing Things Multiple Times

**Basic Structure:**
```python
for variable in sequence:
    # Do something with variable
```

**Example 1: Counting Stars**
```python
for star in range(5):
    print("⭐" * (star + 1))
```

**Output:**
```
⭐
⭐⭐
⭐⭐⭐
⭐⭐⭐⭐
⭐⭐⭐⭐⭐
```

**Example 2: Greeting Friends**
```python
friends = ["Alice", "Bob", "Charlie", "Diana"]

for friend in friends:
    print(f"Hello, {friend}! 👋")
```

**The range() Function:**
- `range(5)` → 0, 1, 2, 3, 4
- `range(2, 7)` → 2, 3, 4, 5, 6
- `range(0, 10, 2)` → 0, 2, 4, 6, 8

### The while Loop: Keep Going Until...

**Structure:**
```python
while condition:
    # Do something
    # Update condition (important!)
```

**Example: Countdown Timer**
```python
countdown = 5

while countdown > 0:
    print(f"🚀 {countdown}...")
    countdown = countdown - 1

print("🎉 Blast off!")
```

**⚠️ Danger Zone: Infinite Loops**
```python
# DON'T DO THIS!
while True:
    print("This will run forever!")
```

**How to avoid:** Always make sure your condition can become False!

### Loop Control Statements

**break: Emergency Exit**
```python
for number in range(1, 11):
    if number == 7:
        print("Lucky number 7 found! 🍀")
        break
    print(f"Number: {number}")
```

**continue: Skip This One**
```python
for number in range(1, 6):
    if number == 3:
        print("Skipping number 3!")
        continue
    print(f"Processing number: {number}")
```

**Fun Example: Guessing Game**
```python
secret_number = 7
guess = 0

while guess != secret_number:
    guess = int(input("Guess a number between 1-10: "))
    
    if guess < secret_number:
        print("Too low! 📉")
    elif guess > secret_number:
        print("Too high! 📈")
    else:
        print("Perfect! You got it! 🎯")
```

---

## Functions for Code Organization

### What is a Function and Why Use One?

**Think of a function like a recipe:**
- Input: Ingredients (parameters)
- Process: Cooking steps (function body)
- Output: Delicious meal (return value)

**Benefits:**
- **Reusability:** Write once, use many times
- **Readability:** Code becomes easier to understand
- **Abstraction:** Hide complex details

### Defining a Function: The def Keyword

**Basic Structure:**
```python
def function_name():
    # Function body
    pass
```

**Example: Simple Greeting**
```python
def say_hello():
    print("Hello, world! 👋")

# Call the function
say_hello()
```

### Parameters and Arguments: Passing Information In

**Parameters** are like placeholders:
```python
def greet_person(name):
    print(f"Hello, {name}! 😊")

# Arguments are the actual values
greet_person("Alice")
greet_person("Bob")
```

**Multiple Parameters:**
```python
def introduce_friends(name1, name2):
    print(f"Meet {name1} and {name2}! They're best friends! 👫")

introduce_friends("Tom", "Jerry")
```

**Default Parameters:**
```python
def greet_with_time(name, time_of_day="morning"):
    print(f"Good {time_of_day}, {name}! ☀️")

greet_with_time("Alice")  # Uses default "morning"
greet_with_time("Bob", "evening")  # Uses "evening"
```

**Keyword Arguments:**
```python
def order_pizza(size, toppings="cheese", delivery=True):
    print(f"Ordering a {size} pizza with {toppings}")
    if delivery:
        print("Will be delivered! 🚚")

order_pizza("large")
order_pizza(size="medium", toppings="pepperoni")
order_pizza("small", delivery=False, toppings="mushrooms")
```

### The return Statement: Getting Information Out

**Functions can give us back results:**
```python
def add_numbers(a, b):
    result = a + b
    return result

sum_result = add_numbers(5, 3)
print(f"5 + 3 = {sum_result}")
```

**Fun Example: Temperature Converter**
```python
def celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5/9
    return celsius

# Usage
temp_c = 25
temp_f = celsius_to_fahrenheit(temp_c)
print(f"{temp_c}°C = {temp_f}°F")
```

### Docstrings: Documenting Your Functions

**Good practice - explain what your function does:**
```python
def calculate_area(length, width):
    """
    Calculate the area of a rectangle.
    
    Parameters:
    length (float): The length of the rectangle
    width (float): The width of the rectangle
    
    Returns:
    float: The area of the rectangle
    """
    return length * width
```

### Variable Scope: Local vs Global

**Local Variables:** Live inside the function
```python
def my_function():
    local_var = "I'm local!"
    print(local_var)

my_function()
# print(local_var)  # This would cause an error!
```

**Global Variables:** Live everywhere
```python
global_var = "I'm global!"

def my_function():
    print(global_var)  # This works!

my_function()
print(global_var)  # This also works!
```

---

## Basic Error Handling

### The Problem: When Things Go Wrong

**Common crashes:**
```python
# Division by zero
result = 10 / 0  # ZeroDivisionError!

# Invalid conversion
age = int("hello")  # ValueError!

# Accessing non-existent item
numbers = [1, 2, 3]
print(numbers[10])  # IndexError!
```

### The Solution: try...except Block

**Basic Structure:**
```python
try:
    # Code that might cause an error
    pass
except:
    # Code that runs if an error occurs
    pass
```

**Example: Safe Division**
```python
try:
    num1 = 10
    num2 = 0
    result = num1 / num2
    print(f"Result: {result}")
except:
    print("Oops! Something went wrong with the division! 🤔")
```

### Practical Example: User Input Validation

**Without error handling:**
```python
age = int(input("Enter your age: "))  # Crashes if user enters "hello"
```

**With error handling:**
```python
try:
    age = int(input("Enter your age: "))
    if age >= 18:
        print("You can vote! 🗳️")
    else:
        print("Not old enough to vote yet! 👶")
except:
    print("Please enter a valid number! 🔢")
```

### Specific Error Types

**Catching specific errors:**
```python
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Cannot divide by zero! ⚠️")
        return None
    except ValueError:
        print("Invalid numbers provided! 🚫")
        return None

print(safe_divide(10, 2))  # Works fine
print(safe_divide(10, 0))  # Handles division by zero
```

**Real-world example: Age Validator**
```python
def get_valid_age():
    while True:
        try:
            age = int(input("Enter your age: "))
            if age < 0:
                print("Age cannot be negative! 👶")
                continue
            if age > 150:
                print("That's quite old! Are you sure? 👴")
                continue
            return age
        except ValueError:
            print("Please enter a valid number! 🔢")

user_age = get_valid_age()
print(f"Your age is: {user_age}")
```

---

## Common Error Types: Quick Reference

Here are the most common errors you'll encounter in Python:

```python
# ValueError - Invalid value conversion
age = int("hello")  # Can't convert "hello" to integer

# ZeroDivisionError - Division by zero
result = 10 / 0  # Mathematical impossibility

# IndexError - Accessing non-existent list/string position
numbers = [1, 2, 3]
print(numbers[10])  # Index 10 doesn't exist

# KeyError - Accessing non-existent dictionary key
person = {"name": "Alice", "age": 25}
print(person["height"])  # Key "height" doesn't exist

# TypeError - Wrong data type operation
result = "5" + 3  # Can't add string and integer directly

# NameError - Using undefined variable
print(undefined_variable)  # Variable doesn't exist

# FileNotFoundError - File doesn't exist
file = open("missing_file.txt")  # File not found

# AttributeError - Method/attribute doesn't exist
text = "hello"
text.append("world")  # Strings don't have append method
```

**Pro Tip:** Always use specific error types in your `except` blocks when possible - it makes debugging much easier!

---

## Wrap-up

### Summary of Topics Covered

**🎯 Today's Journey:**

1. **Control Flow** - Made our programs smart and decision-capable
2. **Conditional Statements** - `if`, `elif`, `else` for making choices
3. **Loops** - `for` and `while` for repetition, with `break` and `continue`
4. **Functions** - `def`, parameters, `return` for code organization
5. **Error Handling** - `try...except` for graceful failure management

**🚀 Key Takeaways:**
- Indentation is crucial in Python!
- Functions make code reusable and readable
- Always handle potential errors gracefully
- Control flow transforms linear code into intelligent programs

**🎮 Final Challenge:**
Can you create a simple number guessing game that:
- Uses a function to generate a random number
- Uses a while loop to keep asking for guesses
- Uses if/elif/else to give hints
- Uses try/except to handle invalid input

**💡 Solution:**
```python
import random

def generate_random_number():
    """Generate a random number between 1 and 10"""
    return random.randint(1, 10)

def number_guessing_game():
    """Main game function that combines all control flow concepts"""
    secret_number = generate_random_number()
    attempts = 0
    
    print("🎯 Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 10...")
    
    while True:
        try:
            guess = int(input("Enter your guess: "))
            attempts += 1
            
            if guess < secret_number:
                print("📉 Too low! Try again.")
            elif guess > secret_number:
                print("📈 Too high! Try again.")
            else:
                print(f"🎉 Congratulations! You got it in {attempts} attempts!")
                break
                
        except ValueError:
            print("🚫 Please enter a valid number!")

# Start the game
number_guessing_game()
```

**What this solution demonstrates:**
- ✅ **Function**: `generate_random_number()` and `number_guessing_game()`
- ✅ **While loop**: Continues until correct guess
- ✅ **If/elif/else**: Provides hints (too low, too high, correct)
- ✅ **Try/except**: Handles invalid input gracefully
- ✅ **Break statement**: Exits loop when game is won

**Remember:** Programming is like learning to drive - practice makes perfect! 🚗💨

---

### Quick Reference Card

```python
# Conditionals
if condition:
    # do something
elif other_condition:
    # do something else
else:
    # default action

# Loops
for item in sequence:
    # repeat for each item

while condition:
    # repeat while true

# Functions
def function_name(parameter1, parameter2="default"):
    """Docstring explaining the function"""
    # function body
    return result

# Error Handling
try:
    # risky code
except SpecificError:
    # handle specific error
except:
    # handle any other error
```

*Happy coding! 🐍✨*

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*