**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Python Data Structures: Dictionaries and Sets

---

## Introduction

Today we're covering two essential Python data structures: dictionaries and sets. If you've been working with lists and tuples, these will give you much more flexibility and efficiency for certain types of problems.

Dictionaries store key-value pairs and give you fast lookups - instead of searching through a list to find something, you can access it directly by its key. Sets are collections of unique items, which makes them perfect for removing duplicates and checking if something exists in a collection.

---

## Dictionaries Deep Dive

### What Makes Dictionaries Special?

Dictionaries are Python's implementation of hash tables—one of computer science's most elegant data structures. They provide O(1) average-case lookup time, making them incredibly efficient for data retrieval.

```python
# Creating dictionaries - multiple ways
student = {'name': 'Alice', 'age': 22, 'major': 'Computer Science'}
student_alt = dict(name='Alice', age=22, major='Computer Science')
student_from_pairs = dict([('name', 'Alice'), ('age', 22)])

print(student)  # {'name': 'Alice', 'age': 22, 'major': 'Computer Science'}
```

### Essential Dictionary Operations

```python
# Basic operations
student = {'name': 'Alice', 'age': 22, 'major': 'Computer Science'}

# Accessing values
print(student['name'])          # 'Alice'
print(student.get('name'))      # 'Alice'
print(student.get('grade', 'Not found'))  # 'Not found' (default value)

# Adding/updating
student['grade'] = 'A'
student.update({'semester': 4, 'gpa': 3.8})

# Removing items
del student['age']              # Remove specific key
grade = student.pop('grade')    # Remove and return value
last_item = student.popitem()   # Remove and return arbitrary key-value pair

print(student)
```

### Essential Dictionary Methods

```python
# The power trio: keys(), values(), items()
inventory = {'apples': 50, 'bananas': 30, 'oranges': 25}

# Iterating through keys
for fruit in inventory.keys():
    print(f"We have {fruit}")

# Iterating through values
for quantity in inventory.values():
    print(f"Quantity: {quantity}")

# Iterating through key-value pairs (most common!)
for fruit, quantity in inventory.items():
    print(f"{fruit}: {quantity}")

# Pro tip: You can convert these to lists if needed
fruit_list = list(inventory.keys())
```

### Dictionary Comprehensions

```python
# Basic dictionary comprehension
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# With conditions
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)  # {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}

# Processing existing data
temperatures_c = {'London': 15, 'New York': 20, 'Tokyo': 25}
temperatures_f = {city: (temp * 9/5) + 32 for city, temp in temperatures_c.items()}
print(temperatures_f)  # {'London': 59.0, 'New York': 68.0, 'Tokyo': 77.0}
```



### Nested Dictionaries: Handling Complex Data

```python
# Real-world example: Student database
students_db = {
    'CS101': {
        'students': ['Alice', 'Bob', 'Charlie'],
        'instructor': 'Dr. Smith',
        'credits': 3
    },
    'MATH201': {
        'students': ['Alice', 'Diana', 'Eve'],
        'instructor': 'Dr. Johnson',
        'credits': 4
    }
}

# Safely accessing nested data
def get_instructor(course_code):
    course = students_db.get(course_code)
    if course:
        return course.get('instructor', 'Unknown')
    return 'Course not found'

print(get_instructor('CS101'))  # Dr. Smith
```

---

## Sets Mastery

### Understanding Sets: The Unordered Collection

Sets are collections of unique elements. Think of them as mathematical sets—no duplicates allowed, and order doesn't matter.

```python
# Creating sets
fruits = {'apple', 'banana', 'orange'}
numbers = set([1, 2, 3, 4, 5])
empty_set = set()  # Note: {} creates an empty dict, not set!

# Sets automatically remove duplicates
duplicates = {1, 2, 2, 3, 3, 3, 4}
print(duplicates)  # {1, 2, 3, 4}

# Converting lists to sets (duplicate removal)
messy_list = [1, 2, 2, 3, 3, 3, 4, 4]
clean_set = set(messy_list)
back_to_list = list(clean_set)
print(back_to_list)  # [1, 2, 3, 4]
```

### Set Operations

```python
# Set operations - the mathematical way
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}

# Union (all elements from both sets)
union = set_a | set_b
print(union)  # {1, 2, 3, 4, 5, 6, 7, 8}

# Intersection (common elements)
intersection = set_a & set_b
print(intersection)  # {4, 5}

# Difference (elements in set_a but not in set_b)
difference = set_a - set_b
print(difference)  # {1, 2, 3}

# Symmetric difference (elements in either set, but not both)
sym_diff = set_a ^ set_b
print(sym_diff)  # {1, 2, 3, 6, 7, 8}
```

### Set Methods and Practical Applications

```python
# Modifying sets
skills = {'Python', 'JavaScript', 'SQL'}

# Adding elements
skills.add('React')
skills.update(['Docker', 'AWS'])  # Add multiple elements

# Removing elements
skills.remove('JavaScript')  # Raises KeyError if not found
skills.discard('PHP')        # Doesn't raise error if not found

print(skills)

# Membership testing (super fast!)
print('Python' in skills)  # True
print('Java' in skills)    # False

# Set comparisons
basic_skills = {'Python', 'SQL'}
advanced_skills = {'Python', 'JavaScript', 'React', 'Docker'}

print(basic_skills.issubset(advanced_skills))    # True
print(advanced_skills.issuperset(basic_skills))  # True
```

### Real-World Set Applications

```python
# Example 1: Finding unique visitors across multiple days
monday_visitors = {'Alice', 'Bob', 'Charlie', 'Diana'}
tuesday_visitors = {'Bob', 'Charlie', 'Eve', 'Frank'}
wednesday_visitors = {'Alice', 'Charlie', 'Frank', 'Grace'}

# Total unique visitors
all_visitors = monday_visitors | tuesday_visitors | wednesday_visitors
print(f"Total unique visitors: {len(all_visitors)}")

# Visitors who came all three days
consistent_visitors = monday_visitors & tuesday_visitors & wednesday_visitors
print(f"Consistent visitors: {consistent_visitors}")

# Visitors who came only on Monday
monday_only = monday_visitors - tuesday_visitors - wednesday_visitors
print(f"Monday-only visitors: {monday_only}")
```

### Set Comprehensions

```python
# Basic set comprehension
squares = {x**2 for x in range(1, 6)}
print(squares)  # {1, 4, 9, 16, 25}

# With conditions
even_squares = {x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)  # {4, 16, 36, 64, 100}

# Practical example: extracting unique domains from email list
emails = ['user1@gmail.com', 'user2@yahoo.com', 'user3@gmail.com', 'user4@hotmail.com']
domains = {email.split('@')[1] for email in emails}
print(domains)  # {'gmail.com', 'yahoo.com', 'hotmail.com'}
```

---

## Advanced Techniques

### Dictionary as a Database

```python
# Creating a simple in-memory database
class SimpleDB:
    def __init__(self):
        self.users = {}
        self.next_id = 1
    
    def add_user(self, name, email):
        user_id = self.next_id
        self.users[user_id] = {'name': name, 'email': email}
        self.next_id += 1
        return user_id
    
    def get_user(self, user_id):
        return self.users.get(user_id)
    
    def find_by_email(self, email):
        for user_id, user_data in self.users.items():
            if user_data['email'] == email:
                return user_id, user_data
        return None

# Usage
db = SimpleDB()
id1 = db.add_user('Alice', 'alice@example.com')
id2 = db.add_user('Bob', 'bob@example.com')

print(db.get_user(id1))  # {'name': 'Alice', 'email': 'alice@example.com'}
```

### Set-Based Algorithms

```python
# Finding common friends (social network style)
def find_mutual_friends(user1_friends, user2_friends):
    return user1_friends & user2_friends

# Finding suggested friends
def suggest_friends(user_friends, all_friends_of_friends):
    # Friends of friends, minus current friends and the user themselves
    return all_friends_of_friends - user_friends

# Example usage
alice_friends = {'Bob', 'Charlie', 'Diana'}
bob_friends = {'Alice', 'Charlie', 'Eve', 'Frank'}

mutual = find_mutual_friends(alice_friends, bob_friends)
print(f"Mutual friends: {mutual}")  # {'Charlie'}
```

---

## Real-World Applications

### Web Development Example: Request Processing

```python
# HTTP request data processing
def process_requests(requests):
    # Track unique IPs
    unique_ips = set()
    
    # Count requests per endpoint
    endpoint_counts = {}
    
    # Track user agents
    user_agents = set()
    
    for request in requests:
        # Add IP to set (automatically handles duplicates)
        unique_ips.add(request['ip'])
        
        # Count endpoints
        endpoint = request['endpoint']
        endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        
        # Track user agents
        user_agents.add(request['user_agent'])
    
    return {
        'unique_visitors': len(unique_ips),
        'endpoint_stats': endpoint_counts,
        'browser_diversity': len(user_agents)
    }

# Sample data
requests = [
    {'ip': '192.168.1.1', 'endpoint': '/home', 'user_agent': 'Chrome'},
    {'ip': '192.168.1.2', 'endpoint': '/about', 'user_agent': 'Firefox'},
    {'ip': '192.168.1.1', 'endpoint': '/contact', 'user_agent': 'Chrome'},
    {'ip': '192.168.1.3', 'endpoint': '/home', 'user_agent': 'Safari'},
]

stats = process_requests(requests)
print(stats)
```




## Performance Considerations

### When to Use What

```python
import time

# Demonstration of lookup performance
def performance_test():
    # Create test data
    data_list = list(range(100000))
    data_set = set(data_list)
    data_dict = {i: f"value_{i}" for i in data_list}
    
    search_value = 99999
    
    # List lookup (O(n))
    start = time.time()
    result = search_value in data_list
    list_time = time.time() - start
    
    # Set lookup (O(1) average)
    start = time.time()
    result = search_value in data_set
    set_time = time.time() - start
    
    # Dict lookup (O(1) average)
    start = time.time()
    result = search_value in data_dict
    dict_time = time.time() - start
    
    print(f"List lookup: {list_time:.6f}s")
    print(f"Set lookup: {set_time:.6f}s")
    print(f"Dict lookup: {dict_time:.6f}s")

# Run the test
performance_test()
```

---


### Best Practice: Use get() with Default Values

```python
# Instead of this:
def process_config(config):
    if 'timeout' in config:
        timeout = config['timeout']
    else:
        timeout = 30
    
    if 'retries' in config:
        retries = config['retries']
    else:
        retries = 3

# Do this:
def process_config(config):
    timeout = config.get('timeout', 30)
    retries = config.get('retries', 3)
```

### Best Practice: Use Sets for Membership Testing

```python
# Instead of this (slow for large lists):
valid_ids = [100, 200, 300, 400, 500]
if user_id in valid_ids:
    process_user()

# Do this (much faster):
valid_ids = {100, 200, 300, 400, 500}
if user_id in valid_ids:
    process_user()
```

---

## Conclusion

Dictionaries and sets are not just data structures—they're problem-solving tools that can make your code more efficient, readable, and Pythonic. Here's what we've covered:

**Dictionaries are perfect for:**
- Storing key-value relationships
- Fast lookups and data retrieval
- Counting and grouping operations


**Sets are ideal for:**
- Removing duplicates
- Membership testing
- Mathematical set operations
- Finding unique elements
- Comparing collections

**Key Takeaways:**
1. Use dictionaries when you need fast key-based access to data
2. Use sets when you need unique collections and fast membership testing
3. Leverage comprehensions for concise, readable code
4. Consider performance implications in your design choices
5. Always think about the time complexity of your operations

Remember: the best Python developers don't just know the syntax—they understand when and why to use each tool. Practice these concepts, experiment with the examples, and soon you'll be leveraging the full power of Python's collections in your own projects.

---

*Happy coding, and may your dictionaries always have the keys you're looking for! 🐍*

---
**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*