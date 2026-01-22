**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Object-Oriented Programming in Python
## From Basics to Building Real Applications

---

## 1. Introduction to OOP

### What is Object-Oriented Programming?

**Object-Oriented Programming (OOP)** is like organizing your code the way we organize things in real life. Instead of writing everything in one big file, we create **classes** that represent real-world things.

Think about it this way:
- A **class** is like a **blueprint** or template
- An **object** is like a **house built from that blueprint**

### Why Use OOP?

1. **Code Reusability** - Write once, use many times
2. **Organization** - Keep related code together
3. **Real-world modeling** - Code that makes sense
4. **Easier maintenance** - Fix bugs in one place

### The Four Pillars of OOP

1. **Encapsulation** - Keeping data safe inside classes
2. **Inheritance** - Child classes getting features from parent classes
3. **Polymorphism** - Same method name, different behaviors
4. **Abstraction** - Hiding complex details

### A Simple Example: Why OOP Makes Life Easier

Let's see the difference between **procedural** (old way) and **object-oriented** (better way) code:

#### ❌ Without OOP - Everything Mixed Together

### Using Properties for Even Better Control

```python
# calculator_messy.py - The old, confusing way

# Variables scattered everywhere
num1 = 0
num2 = 0
result = 0
history = []

def add_numbers():
    global num1, num2, result, history
    result = num1 + num2
    history.append(f"{num1} + {num2} = {result}")
    return result

def subtract_numbers():
    global num1, num2, result, history
    result = num1 - num2
    history.append(f"{num1} - {num2} = {result}")
    return result

def show_history():
    global history
    for calculation in history:
        print(calculation)

def clear_history():
    global history
    history = []

# Using it (confusing!)
num1 = 10
num2 = 5
add_numbers()
print(f"Result: {result}")  # Wait, which result?

num1 = 20  # Oops, accidentally changed global variable
subtract_numbers()
print(f"Result: {result}")  # Now I'm confused!
```

#### ✅ With OOP - Clean and Organized

```python
# calculator_clean.py - The OOP way

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, num1, num2):
        result = num1 + num2
        self.history.append(f"{num1} + {num2} = {result}")
        return result
    
    def subtract(self, num1, num2):
        result = num1 - num2
        self.history.append(f"{num1} - {num2} = {result}")
        return result
    
    def show_history(self):
        for calculation in self.history:
            print(calculation)
    
    def clear_history(self):
        self.history = []

# Using it (much clearer!)
my_calc = Calculator()
result1 = my_calc.add(10, 5)
result2 = my_calc.subtract(20, 3)

print(f"First result: {result1}")   # Clear and separate
print(f"Second result: {result2}")  # No confusion!

my_calc.show_history()  # See what we did
```

### Why the OOP Version is Better

1. **No Global Variables** - Everything stays organized inside the class
2. **Clear Ownership** - Each calculator has its own history
3. **Easy to Use** - `my_calc.add(5, 3)` is much clearer
4. **No Accidents** - Can't accidentally mess up someone else's calculator
5. **Reusable** - Can create multiple calculators easily

```python
# Multiple calculators - impossible with the messy version!
work_calc = Calculator()
personal_calc = Calculator()

work_calc.add(100, 200)
personal_calc.add(5, 10)

# Each keeps its own history - no mixing up!
```

**This is the power of OOP** - it takes messy, confusing code and makes it **clean, organized, and logical**!

---

## 2. Creating Your First Class

Let's start with something everyone can understand - a **Student**!

### Basic Class Structure

```python
# student.py
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.grades = []
    
    def add_grade(self, grade):
        self.grades.append(grade)
    
    def get_average(self):
        if len(self.grades) == 0:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def introduce(self):
        return f"Hi! I'm {self.name} and I'm {self.age} years old."
```

### Key Components Explained

- **`class Student:`** - This creates our blueprint
- **`__init__`** - This is the **constructor**, it runs when we create a new student
- **`self`** - This refers to the **current object**
- **`self.name`** - This is an **attribute** (property) of the student
- **Methods** - These are **functions** inside the class

### Using Our Student Class

```python
# test_student.py
from student import Student

# Creating objects (instances)
john = Student("John", 20)
mary = Student("Mary", 19)

# Using methods
print(john.introduce())  # Hi! I'm John and I'm 20 years old.
print(mary.introduce())  # Hi! I'm Mary and I'm 19 years old.

# Adding grades
john.add_grade(85)
john.add_grade(92)
john.add_grade(78)

print(f"John's average: {john.get_average()}")  # John's average: 85.0
```

### 🎯 Practical Exercise 1
Create a `Book` class with:
- Attributes: title, author, pages
- Methods: get_info(), mark_as_read()

#### Solution:

```python
# book_exercise.py
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
        self.is_read = False  # Track reading status
    
    def get_info(self):
        status = "Read" if self.is_read else "Not read yet"
        return f"'{self.title}' by {self.author} - {self.pages} pages ({status})"
    
    def mark_as_read(self):
        self.is_read = True
        return f"You finished reading '{self.title}'!"

# Test the Book class
book1 = Book("Python Basics", "John Smith", 250)
book2 = Book("Web Development", "Jane Doe", 400)

print(book1.get_info())  # 'Python Basics' by John Smith - 250 pages (Not read yet)
print(book1.mark_as_read())  # You finished reading 'Python Basics'!
print(book1.get_info())  # 'Python Basics' by John Smith - 250 pages (Read)
```

---

## 3. Understanding Objects and Methods

### Objects are Independent

Each **object** has its own **data**. When we create two students, they don't share grades:

```python
# Different students, different data
student1 = Student("Alice", 20)
student2 = Student("Bob", 21)

student1.add_grade(90)
student2.add_grade(75)

print(student1.get_average())  # 90.0
print(student2.get_average())  # 75.0
```

### Method Types

Understanding the **three types of methods** in Python classes:

1. **Instance Methods** - Work with **specific object data** (most common)
   - Use `self` to access object's data
   - Each object can have different results
   - Example: `student.get_average()` - each student has different grades

2. **Class Methods** - Work with **class-level data** (shared by all objects)
   - Use `@classmethod` decorator and `cls` parameter
   - Same result for all objects of that class
   - Example: `Student.get_school_info()` - same school for all students

3. **Static Methods** - **Don't need object or class data** (utility functions)
   - Use `@staticmethod` decorator, no `self` or `cls`
   - Work like regular functions but belong to the class
   - Example: `Student.is_passing_grade(75)` - just checks if 75 is passing

```python
class Student:
    # Class variables - shared by ALL students
    school_name = "Python University"
    total_students = 0
    
    def __init__(self, name, age):
        self.name = name  # Instance variable - unique to each student
        self.age = age    # Instance variable - unique to each student
        self.grades = []  # Instance variable - unique to each student
        Student.total_students += 1  # Update class variable
    
    # INSTANCE METHOD - works with THIS student's data
    def add_grade(self, grade):
        """Add a grade to THIS student's grade list"""
        self.grades.append(grade)  # Uses self - specific to this student
        return f"{self.name} received grade: {grade}"
    
    # INSTANCE METHOD - works with THIS student's data
    def get_average(self):
        """Calculate THIS student's average"""
        if len(self.grades) == 0:
            return 0
        return sum(self.grades) / len(self.grades)  # Uses self.grades
    
    # CLASS METHOD - works with data shared by ALL students
    @classmethod
    def get_school_info(cls):
        """Get information about the school - same for ALL students"""
        return f"School: {cls.school_name}, Total Students: {cls.total_students}"
    
    # STATIC METHOD - doesn't need ANY student data
    @staticmethod
    def is_passing_grade(grade):
        """Check if a grade is passing - works without any student object"""
        return grade >= 60  # Just a utility function
    
    # STATIC METHOD - doesn't need ANY student data  
    @staticmethod
    def grade_letter(grade):
        """Convert number grade to letter - works without any student object"""
        if grade >= 90: return 'A'
        elif grade >= 80: return 'B'
        elif grade >= 70: return 'C'
        elif grade >= 60: return 'D'
        else: return 'F'

# Let's see the differences in action!

# Create some students (each has their own data)
john = Student("John", 20)
mary = Student("Mary", 19)

# INSTANCE METHODS - different results for different students
john.add_grade(85)
john.add_grade(92)
mary.add_grade(78)
mary.add_grade(95)

print("=== INSTANCE METHODS ===")
print(f"John's average: {john.get_average()}")  # 88.5
print(f"Mary's average: {mary.get_average()}")  # 86.5 - Different result!

# CLASS METHODS - same result no matter which student calls it
print("\n=== CLASS METHODS ===")
print(john.get_school_info())  # Same school info
print(mary.get_school_info())  # Exactly the same!
print(Student.get_school_info())  # Can call directly on class too!

# STATIC METHODS - work without any student object
print("\n=== STATIC METHODS ===")
print(f"Is 75 passing? {Student.is_passing_grade(75)}")  # True
print(f"Is 45 passing? {Student.is_passing_grade(45)}")  # False
print(f"Grade 87 is: {Student.grade_letter(87)}")       # B

# Static methods can be called from instances too (but don't need to be)
print(f"From John: Is 85 passing? {john.is_passing_grade(85)}")  # True
```

### When to Use Each Type?

**🎯 Use Instance Methods when:**
- You need to work with specific object data
- Each object might give different results
- Examples: `calculate_balance()`, `get_name()`, `add_item()`

**🏫 Use Class Methods when:**
- You need to work with data shared by all objects
- You want to create objects in special ways (factory methods)
- Examples: `get_total_count()`, `create_from_file()`, `get_default_settings()`

**🔧 Use Static Methods when:**
- You have a utility function that belongs with the class
- You don't need any object or class data
- Examples: `is_valid_email()`, `convert_temperature()`, `calculate_tax()`

---

## 4. Encapsulation - Keeping Data and Methods Safe

**Encapsulation** means keeping some data **and methods** **private** so they can't be accidentally used or changed from outside the class.

### The Problem Without Encapsulation

```python
# Bad example - anyone can change the balance directly
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance  # This is public - anyone can change it!

# This is dangerous!
account = BankAccount("John", 1000)
account.balance = -5000  # Oops! Negative balance
```

### The Solution - Private Attributes and Methods

```python
# bank_account.py
class BankAccount:
    def __init__(self, owner, initial_balance):
        self.owner = owner
        self.__balance = initial_balance  # Private attribute (double underscore)
    
    def deposit(self, amount):
        if self.__is_valid_amount(amount):  # Using private method
            self.__balance += amount
            self.__log_transaction(f"Deposited ${amount}")  # Using private method
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        return "Deposit amount must be positive"
    
    def withdraw(self, amount):
        if self.__is_valid_amount(amount) and amount <= self.__balance:
            self.__balance -= amount
            self.__log_transaction(f"Withdrew ${amount}")  # Using private method
            return f"Withdrew ${amount}. New balance: ${self.__balance}"
        return "Invalid withdrawal amount"
    
    def get_balance(self):
        return self.__balance
    
    def get_account_info(self):
        return f"Account Owner: {self.owner}, Balance: ${self.__balance}"
    
    # PRIVATE METHODS - only used inside the class
    def __is_valid_amount(self, amount):
        """Check if amount is valid - internal use only"""
        return amount > 0 and isinstance(amount, (int, float))
    
    def __log_transaction(self, message):
        """Log transaction details - internal use only"""
        print(f"[INTERNAL LOG] {self.owner}: {message}")
    
    def __calculate_interest(self, rate):
        """Calculate interest - complex internal logic"""
        return self.__balance * rate * 0.01

# Usage - shows what's accessible from outside
account = BankAccount("John", 1000)

# ✅ These work - public methods
print(account.deposit(200))
print(account.get_balance())

# ❌ These don't work - private methods are hidden
# account.__is_valid_amount(100)      # AttributeError!
# account.__log_transaction("test")   # AttributeError!
# account.__calculate_interest(5)     # AttributeError!
```

### Practical: Organizing Your Code into Files

When working on these examples, it's a good practice to create separate Python files for each class or main program. This keeps your code organized and easier to manage. Here’s how you can do it:

- **bank_account.py**: Put your `BankAccount` class in this file.
- **banking_demo.py**: Put your main program (the code that creates accounts and simulates operations) in this file.

To try the example below:
1. Copy the `BankAccount` class code into a file named `bank_account.py`.
2. Copy the demo code into a file named `banking_demo.py` (as shown below).
3. Make sure both files are in the same folder so the import works.
4. Run `banking_demo.py` to see the output.

```python
# banking_demo.py
from bank_account import BankAccount

def main():
    # Create account
    my_account = BankAccount("Alice", 500)
    
    # Simulate banking operations
    print(my_account.get_account_info())
    print(my_account.deposit(200))
    print(my_account.withdraw(100))
    print(my_account.withdraw(1000))  # This should fail
    
if __name__ == "__main__":
    main()
```

### Why Encapsulate Methods Too?

**Private methods** help us:

1. **Hide Complex Logic** - Users don't need to know how interest is calculated
2. **Prevent Misuse** - Can't accidentally call internal validation functions
3. **Keep Code Clean** - Only show methods that users actually need
4. **Easy to Change** - Can modify private methods without breaking user code


---

### Using @property for Attribute Control

Another way to control access to class attributes is by using the `@property` decorator. This allows you to define methods that act like attributes, providing a clean way to get (read) and set (write) values while still keeping control over how the data is accessed or changed.

- The **getter** method lets you read the value (like `account.balance`).
- The **setter** method lets you set the value (like `account.balance = 1000`).

This is useful for validation or hiding internal details.

```python
class BankAccount:
    def __init__(self, owner, initial_balance):
        self.owner = owner
        self.__balance = initial_balance  # Private attribute
    
    @property  # This is the GETTER
    def balance(self):
        """Get the current balance (read-only access)"""
        return self.__balance
    
    @balance.setter  # This is the SETTER
    def balance(self, amount):
        """Set the balance (with validation)"""
        if amount >= 0:
            self.__balance = amount
        else:
            print("Balance cannot be negative!")

# Usage
account = BankAccount("John", 1000)
print(account.balance)  # Calls the getter: 1000
account.balance = 1500  # Calls the setter
account.balance = -100  # This will show error message
```




---

### Mortgage Calculation Example

A **mortgage** is a type of loan used to buy expensive items like a house or a car. For example, if you want to buy a car that costs $20,000 but you only have $5,000, you might take a loan (mortgage) for the remaining $15,000 and pay it back in monthly installments, with interest, over several years. The monthly payment is calculated so that you pay off both the loan and the interest over the agreed period.

The formula for a fixed-rate mortgage (used for both houses and cars) is:

    M = P * [r(1 + r)^n] / [(1 + r)^n - 1]

Where:
- M = monthly payment
- P = principal (loan amount)
- r = monthly interest rate (annual rate / 12 / 100)
- n = total number of payments (years * 12)

---

```python
class Calculator:
    def __init__(self):
        self.history = []
    
    # PUBLIC METHOD - users can call this
    def calculate_mortgage(self, principal, rate, years):
        """
        Calculate monthly mortgage payment.
        
        principal: The amount borrowed (loan amount)
        rate: The annual interest rate (e.g., 5.5 for 5.5%)
        years: The number of years for the loan
        
        Formula used (for fixed-rate mortgage):
            M = P * [r(1 + r)^n] / [(1 + r)^n - 1]
        Where:
            M = monthly payment
            P = principal (loan amount)
            r = monthly interest rate (annual rate / 12 / 100)
            n = total number of payments (years * 12)
        """
        monthly_rate = self.__annual_to_monthly_rate(rate)  # Private method
        n = self.__years_to_payments(years)      # Private method
        
        if monthly_rate == 0:
            # No interest case
            return principal / n
        
        # Mortgage payment formula:
        # payment = principal * (rate * (1 + rate)**n) / ((1 + rate)**n - 1)
        payment = self.__mortgage_formula(principal, monthly_rate, n)  # Private
        self.__add_to_history(f"Mortgage calculation: ${payment:.2f}/month")     # Private
        return payment
    
    # PRIVATE METHODS - internal calculations only
    def __annual_to_monthly_rate(self, annual_rate):
        """Convert annual rate to monthly - internal use only"""
        return annual_rate / 100 / 12
    
    def __years_to_payments(self, years):
        """Convert years to number of payments (n) - internal use only"""
        return years * 12
    
    def __mortgage_formula(self, principal, rate, n):
        """
        Complex mortgage formula - internal use only
        Implements: M = P * [r(1 + r)^n] / [(1 + r)^n - 1]
        """
        return principal * (rate * (1 + rate)**n) / ((1 + rate)**n - 1)
    
    def __add_to_history(self, entry):
        """Add entry to calculation history - internal use only"""
        self.history.append(entry)

# --- Example: Mortgage Calculation ---
# A mortgage is a loan used to buy a house or a car. You pay it back monthly, with interest.
# This example shows how to calculate the monthly payment for a fixed-rate mortgage.
calc = Calculator()
payment = calc.calculate_mortgage(200000, 5.5, 30)
print(f"Monthly payment: ${payment:.2f}")
# All the complex internal stuff is hidden and protected!
```

---




---

## 5. Inheritance - Reusing Code Smartly

### Inheritance in Python

**Inheritance** lets you create a new class (child) that automatically gets the features (attributes and methods) of another class (parent). This helps you reuse code and model real-world relationships (e.g., a Car is a type of Vehicle).

### Basic Inheritance Example

```python
# vehicles.py
class Vehicle:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self.is_running = False
    
    def start(self):
        self.is_running = True
        return f"{self.brand} {self.model} is now running"
    
    def stop(self):
        self.is_running = False
        return f"{self.brand} {self.model} has stopped"
    
    def get_info(self):
        return f"{self.year} {self.brand} {self.model}"

# Child class inherits from Vehicle
class Car(Vehicle):
    def __init__(self, brand, model, year, doors):
        super().__init__(brand, model, year)  # Call parent constructor
        self.doors = doors
    
    def honk(self):
        return "Beep beep!"
    
    def get_info(self):
        parent_info = super().get_info()
        return f"{parent_info} - {self.doors} doors"

class Motorcycle(Vehicle):
    def __init__(self, brand, model, year, engine_size):
        super().__init__(brand, model, year)
        self.engine_size = engine_size
    
    def rev_engine(self):
        return "Vroom vroom!"
    
    def get_info(self):
        parent_info = super().get_info()
        return f"{parent_info} - {self.engine_size}cc engine"
```

### Using Inherited Classes

```python
# vehicle_demo.py
from vehicles import Vehicle, Car, Motorcycle

def main():
    # Create different vehicles
    my_car = Car("Toyota", "Camry", 2023, 4)
    my_bike = Motorcycle("Honda", "CBR", 2022, 600)
    
    # All vehicles can start and stop
    print(my_car.start())
    print(my_bike.start())
    
    # Each has unique methods
    print(my_car.honk())
    print(my_bike.rev_engine())
    
    # Overridden methods work differently
    print(my_car.get_info())
    print(my_bike.get_info())

if __name__ == "__main__":
    main()
```

### Types of Inheritance

Python supports three main types of inheritance:

1. **Single Inheritance** - One parent, one child (simplest form)
2. **Multiple Inheritance** - One child, multiple parents (inherits from multiple classes)
3. **Multilevel Inheritance** - Grandparent → Parent → Child (chain of inheritance)

#### 1. Single Inheritance

**Definition:** A child class inherits from only one parent class. This is the simplest and most common form of inheritance.

**Example:**
```python
class Person:
    """Parent class representing a person"""
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    def celebrate_birthday(self):
        self.age += 1
        return f"Happy Birthday! Now I'm {self.age} years old."

class Student(Person):
    """Child class inheriting from Person"""
    def __init__(self, name, age, student_id, major):
        # Call parent constructor using super()
        super().__init__(name, age)
        self.student_id = student_id
        self.major = major
    
    def study(self):
        return f"{self.name} is studying {self.major}."
    
    def get_info(self):
        # Use parent method and extend it
        parent_info = super().introduce()
        return f"{parent_info} My student ID is {self.student_id} and I'm majoring in {self.major}."

# Using Single Inheritance
student = Student("Alice", 20, "S12345", "Computer Science")
print(student.introduce())        # Inherited from Person
print(student.study())            # Student's own method
print(student.get_info())         # Overridden method using super()
print(student.celebrate_birthday()) # Inherited from Person
```

#### 2. Multiple Inheritance

**Definition:** A child class inherits from multiple parent classes. The child class can access methods and attributes from all parent classes.

**Example:**
```python
class Employee:
    """First parent class"""
    def __init__(self, name, employee_id, salary):
        self.name = name
        self.employee_id = employee_id
        self.salary = salary
    
    def work(self):
        return f"{self.name} is working."
    
    def get_salary(self):
        return f"{self.name}'s salary is ${self.salary}"

class Developer:
    """Second parent class"""
    def __init__(self, programming_language):
        self.programming_language = programming_language
    
    def code(self):
        return f"Writing code in {self.programming_language}"
    
    def debug(self):
        return f"Debugging {self.programming_language} code"

class DeveloperEmployee(Employee, Developer):
    """Child class inheriting from both Employee and Developer"""
    def __init__(self, name, employee_id, salary, programming_language):
        # Note: We use explicit parent class calls instead of super() in multiple inheritance
        # because super() follows Method Resolution Order (MRO) which can be complex.
        # Explicit calls make it clear which parent's constructor is being called and
        # ensure both parents are initialized correctly.
        # Initialize Employee parent
        Employee.__init__(self, name, employee_id, salary)
        # Initialize Developer parent
        Developer.__init__(self, programming_language)
    
    def get_info(self):
        return f"{self.name} (ID: {self.employee_id}) is a developer earning ${self.salary} and coding in {self.programming_language}"

# Using Multiple Inheritance
dev_emp = DeveloperEmployee("Bob", "E001", 75000, "Python")
print(dev_emp.work())           # From Employee class
print(dev_emp.code())           # From Developer class
print(dev_emp.debug())          # From Developer class
print(dev_emp.get_salary())     # From Employee class
print(dev_emp.get_info())       # From DeveloperEmployee class
```

#### 3. Multilevel Inheritance

**Definition:** A child class inherits from a parent class, which itself inherits from another class. This creates a chain of inheritance: Grandparent → Parent → Child.

**Example:**
```python
class Animal:
    """Grandparent class"""
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def breathe(self):
        return f"{self.name} is breathing..."
    
    def get_info(self):
        return f"Name: {self.name}, Species: {self.species}"

class Mammal(Animal):
    """Parent class inheriting from Animal"""
    def __init__(self, name, species, has_fur):
        # Call grandparent constructor using super()
        super().__init__(name, species)
        self.has_fur = has_fur
    
    def feed_milk(self):
        return f"{self.name} is feeding milk to babies"
    
    def get_info(self):
        # Call grandparent method and extend it
        parent_info = super().get_info()
        fur_status = "has fur" if self.has_fur else "no fur"
        return f"{parent_info}, {fur_status}"

class Dog(Mammal):
    """Child class inheriting from Mammal (which inherits from Animal)"""
    def __init__(self, name, breed):
        # Call parent (Mammal) constructor using super()
        # Mammal will then call Animal constructor
        super().__init__(name, "Canine", has_fur=True)
        self.breed = breed
    
    def bark(self):
        return f"{self.name} says: Woof! Woof!"
    
    def get_info(self):
        # Call parent (Mammal) method, which calls Animal method
        parent_info = super().get_info()
        return f"{parent_info}, Breed: {self.breed}"

# Using Multilevel Inheritance
my_dog = Dog("Buddy", "Golden Retriever")
print(my_dog.breathe())         # From Animal (grandparent)
print(my_dog.feed_milk())       # From Mammal (parent)
print(my_dog.bark())            # From Dog (child)
print(my_dog.get_info())        # Overridden method using super() chain
```

**Key Points:**
- In multilevel inheritance, `super()` calls the immediate parent, which can then call its parent
- The constructor chain flows: Dog → Mammal → Animal
- Methods can be overridden at any level, and `super()` allows calling the parent's version

---

**Summary:**
- Inheritance helps you reuse code and model real-world relationships.
- Use `super().__init__()` to call the parent constructor.
- Child classes can add new methods or override parent methods.
- Multilevel inheritance lets you build more complex hierarchies.

---

## 6. Polymorphism - Same Name, Different Behavior

**Polymorphism** means **"many forms"**. The same method name can do different things in different classes.

### Polymorphism with Shapes

```python
# shapes.py
import math

class Shape:
    def area(self):
        pass
    
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        return 2 * math.pi * self.radius

class Triangle(Shape):
    def __init__(self, base, height, side1, side2):
        self.base = base
        self.height = height
        self.side1 = side1
        self.side2 = side2
    
    def area(self):
        return 0.5 * self.base * self.height
    
    def perimeter(self):
        return self.base + self.side1 + self.side2
```

### Using Polymorphism

```python
# shape_calculator.py
from shapes import Rectangle, Circle, Triangle

rect = Rectangle(5, 3)
cir = Circle(4)
tri =Triangle(6, 4, 5, 5)

print(rect.area())
print(cir.area())
print(tri.area())

print(rect.perimeter())
print(cir.perimeter())
print(tri.perimeter())
```

### Method Overloading Alternative


🔄 Method Overloading – Simple Concept
Method overloading means defining multiple methods with the same name but different parameters (different number or types of arguments), so the correct method is chosen based on how it's called. It's a form of compile-time polymorphism commonly found in languages like Java or C++. However, in Python, method overloading in the traditional sense is not supported — if you define multiple methods with the same name, the last one will overwrite the previous ones. Instead, in Python, we typically use default arguments or *args/**kwargs to mimic overloading behavior, allowing a single method to handle different types or numbers of inputs.



```python
class Calculator:
    def add(self, *args):
        if len(args) == 2:
            return args[0] + args[1]
        elif len(args) == 3:
            return args[0] + args[1] + args[2]
        else:
            return sum(args)

calc = Calculator()
print(calc.add(5, 3))        # 8
print(calc.add(1, 2, 3))     # 6
print(calc.add(1, 2, 3, 4, 5))  # 15
```

---

## 7. Building a Complete Application

Now let's build a **Library Management System** that uses all OOP concepts!

### Step 1: The Book Class

```python
# book.py
class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_borrowed = False
        self.borrowed_by = None
    
    def borrow(self, member_name):
        if not self.is_borrowed:
            self.is_borrowed = True
            self.borrowed_by = member_name
            return f"'{self.title}' borrowed by {member_name}"
        return f"'{self.title}' is already borrowed"
    
    def return_book(self):
        if self.is_borrowed:
            borrower = self.borrowed_by
            self.is_borrowed = False
            self.borrowed_by = None
            return f"'{self.title}' returned by {borrower}"
        return f"'{self.title}' was not borrowed"
    
    def get_info(self):
        status = "Available" if not self.is_borrowed else f"Borrowed by {self.borrowed_by}"
        return f"'{self.title}' by {self.author} - {status}"
```

### Step 2: The Member Class

```python
# member.py
class Member:
    def __init__(self, name, member_id):
        self.name = name
        self.member_id = member_id
        self.borrowed_books = []
    
    def borrow_book(self, book):
        if len(self.borrowed_books) < 3:  # Max 3 books
            result = book.borrow(self.name)
            if "borrowed by" in result:
                self.borrowed_books.append(book)
            return result
        return "Cannot borrow more than 3 books"
    
    def return_book(self, book):
        if book in self.borrowed_books:
            result = book.return_book()
            self.borrowed_books.remove(book)
            return result
        return "You haven't borrowed this book"
    
    def get_borrowed_books(self):
        if not self.borrowed_books:
            return f"{self.name} has no borrowed books"
        
        books_list = [book.title for book in self.borrowed_books]
        return f"{self.name} has borrowed: {', '.join(books_list)}"
```

### Step 3: The Library Class (Composition)

```python
# library.py
from book import Book
from member import Member

class Library:
    def __init__(self, name):
        self.name = name
        self.books = []
        self.members = []
    
    def add_book(self, title, author, isbn):
        book = Book(title, author, isbn)
        self.books.append(book)
        return f"Added '{title}' to library"
    
    def add_member(self, name, member_id):
        member = Member(name, member_id)
        self.members.append(member)
        return f"Added member: {name}"
    
    def find_book(self, title):
        for book in self.books:
            if book.title.lower() == title.lower():
                return book
        return None
    
    def find_member(self, member_id):
        for member in self.members:
            if member.member_id == member_id:
                return member
        return None
    
    def show_all_books(self):
        if not self.books:
            return "No books in library"
        
        result = f"\n=== {self.name} Library Books ===\n"
        for book in self.books:
            result += book.get_info() + "\n"
        return result
    
    def show_available_books(self):
        available = [book for book in self.books if not book.is_borrowed]
        
        if not available:
            return "No books available"
        
        result = f"\n=== Available Books ===\n"
        for book in available:
            result += book.get_info() + "\n"
        return result
```

### Step 4: Main Application

```python
# main.py
from library import Library

def main():
    # Create library
    lib = Library("City Central")
    
    # Add some books
    lib.add_book("Python Programming", "John Smith", "123456789")
    lib.add_book("Web Development", "Jane Doe", "987654321")
    lib.add_book("Data Science", "Bob Wilson", "456789123")
    
    # Add members
    lib.add_member("Alice Johnson", "M001")
    lib.add_member("Charlie Brown", "M002")
    
    # Show all books
    print(lib.show_all_books())
    
    # Find members and books
    alice = lib.find_member("M001")
    python_book = lib.find_book("Python Programming")
    web_book = lib.find_book("Web Development")
    
    # Borrow books
    if alice and python_book:
        print(alice.borrow_book(python_book))
        print(alice.borrow_book(web_book))
    
    # Show member's books
    if alice:
        print(alice.get_borrowed_books())
    
    # Show available books
    print(lib.show_available_books())
    
    # Return a book
    if alice and python_book:
        print(alice.return_book(python_book))
    
    # Show available books again
    print(lib.show_available_books())

if __name__ == "__main__":
    main()
```

---

## 8. Best Practices and Project Structure

### Project Structure

```
library_project/
│
├── main.py           # Main application
├── library.py        # Library class
├── book.py          # Book class
├── member.py        # Member class
├── utils.py         # Helper functions
└── README.md        # Project description
```

### Best Practices

1. **Use Clear Names**
   ```python
   # Good
   class BankAccount:
       def calculate_interest(self):
           pass
   
   # Bad
   class BA:
       def calc_int(self):
           pass
   ```

2. **Keep Methods Small**
   ```python
   # Good - one responsibility per method
   def validate_email(self, email):
       return "@" in email and "." in email
   
   def send_welcome_email(self, email):
       if self.validate_email(email):
           # send email logic
           pass
   ```

3. **Use Docstrings**
   ```python
   class Student:
       """Represents a student with grades and personal information."""
       
       def calculate_gpa(self):
           """Calculate and return the student's GPA based on grades."""
           pass
   ```

4. **Follow the Single Responsibility Principle**
   - Each class should do **one thing well**
   - Each method should have **one clear purpose**



### Common Mistakes to Avoid

1. **Making everything public**
2. **Creating god classes** (classes that do too much)
3. **Not using inheritance when it makes sense**
5. **Forgetting to call `super().__init__()`** in child classes

### Testing Your Code

```python
# test_library.py
def test_book_borrowing():
    from book import Book
    
    book = Book("Test Book", "Test Author", "123")
    result = book.borrow("John")
    
    assert book.is_borrowed == True
    assert book.borrowed_by == "John"
    assert "borrowed by John" in result
    
    print("✅ Book borrowing test passed!")

def test_member_book_limit():
    from member import Member
    from book import Book
    
    member = Member("Test Member", "M999")
    
    # Try to borrow 4 books (limit is 3)
    for i in range(4):
        book = Book(f"Book {i}", "Author", f"ISBN{i}")
        result = member.borrow_book(book)
        
        if i < 3:
            assert "borrowed by" in result
        else:
            assert "Cannot borrow more than 3" in result
    
    print("✅ Member book limit test passed!")

if __name__ == "__main__":
    test_book_borrowing()
    test_member_book_limit()
    print("All tests passed! 🎉")
```

---

## Summary and Key Takeaways

### What We've Learned

1. **Classes and Objects** - How to create blueprints and instances
2. **Encapsulation** - Protecting data with private attributes
3. **Inheritance** - Reusing code through parent-child relationships
4. **Polymorphism** - Same method names, different behaviors

### Real-World Applications

- **Web Applications** - User classes, Product classes
- **Games** - Player classes, Enemy classes, Weapon classes
- **Business Software** - Employee classes, Order classes
- **Mobile Apps** - Profile classes, Message classes

### Next Steps

1. **Practice** building more classes
2. **Learn about** abstract base classes
3. **Explore** design patterns (Factory, Observer, etc.)
4. **Study** frameworks like Django (uses lots of OOP)
5. **Build** your own projects using these concepts

### Quick Reference

```python
# Class template
class MyClass:
    # Class variable
    class_var = "shared by all"
    
    def __init__(self, param):
        self.param = param      # Public
        self._protected = None  # Protected (convention)
        self.__private = None   # Private
    
    def public_method(self):
        return "Anyone can call this"
    
    def _protected_method(self):
        return "Convention: internal use"
    
    def __private_method(self):
        return "Hard to access from outside"
    
    @property
    def param(self):
        return self._param
    
    @param.setter
    def param(self, value):
        if value > 0:
            self._param = value

# Inheritance
class ChildClass(MyClass):
    def __init__(self, param, extra):
        super().__init__(param)
        self.extra = extra
    
    def public_method(self):  # Override
        return "Child version"
```

**Remember**: OOP is about **modeling real-world problems** in code. Start simple, practice often, and gradually build more complex applications!

---

### 🎯 Final Challenge

Create a simple **School Management System** with:
- `Person` base class
- `Student` and `Teacher` child classes
- `Course` class
- `School` class that manages everything

This will test everything you've learned! Start with the basic structure and gradually add features.

**Happy Coding!** 🐍✨