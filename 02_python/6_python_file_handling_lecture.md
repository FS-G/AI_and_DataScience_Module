**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Python File Handling Lecture

## What is File Handling?

File handling in Python means working with files on your computer - reading from them, writing to them, or creating new ones. Think of it like opening a book to read, writing in a notebook, or creating a new document.

## Why Do We Need File Handling?

- Store data permanently (unlike variables that disappear when program ends)
- Read data from external sources
- Save program output for later use
- Work with configuration files, logs, or data files

## Opening Files - The `open()` Function

To work with any file, you first need to open it:

```python
file = open("filename.txt", "mode")
```

### File Modes

- `"r"` - Read mode (default) - opens file for reading
- `"w"` - Write mode - creates new file or overwrites existing one
- `"a"` - Append mode - adds to the end of existing file
- `"x"` - Create mode - creates new file, fails if file exists

## Reading Files

### Method 1: Read Everything at Once

```python
# Open and read entire file
file = open("story.txt", "r")
content = file.read()
print(content)
file.close()  # Always close when done!
```

### Method 2: Read Line by Line

```python
# Read one line at a time
file = open("names.txt", "r")
for line in file:
    print(line.strip())  # strip() removes newline characters
file.close()
```

### Method 3: Read All Lines into a List

```python
file = open("shopping.txt", "r")
lines = file.readlines()
print(lines)  # Each line is an item in the list
file.close()
```

## Writing Files

### Creating a New File

```python
# Write to a new file (overwrites if exists)
file = open("output.txt", "w")
file.write("Hello, World!\n")
file.write("This is my first file.\n")
file.close()
```

### Adding to an Existing File

```python
# Append to existing file
file = open("diary.txt", "a")
file.write("Today was a good day.\n")
file.close()
```

## The Better Way - Using `with` Statement

Instead of manually closing files, use `with` - it automatically closes the file for you:

```python
# Reading with 'with'
with open("data.txt", "r") as file:
    content = file.read()
    print(content)
# File is automatically closed here!

# Writing with 'with'
with open("results.txt", "w") as file:
    file.write("Test results: PASSED")
```

## Complete Examples

### Example 1: Reading a Text File

```python
# Let's say we have a file called "quotes.txt"
with open("quotes.txt", "r") as file:
    quotes = file.readlines()
    
print("Here are today's inspirational quotes:")
for i, quote in enumerate(quotes, 1):
    print(f"{i}. {quote.strip()}")
```

### Example 2: Writing User Input to File

```python
# Get user input and save to file
name = input("What's your name? ")
age = input("How old are you? ")

with open("user_info.txt", "w") as file:
    file.write(f"Name: {name}\n")
    file.write(f"Age: {age}\n")
    
print("Your information has been saved!")
```

### Example 3: Reading and Processing Data

```python
# Read numbers from file and calculate average
with open("numbers.txt", "r") as file:
    numbers = []
    for line in file:
        number = float(line.strip())
        numbers.append(number)

if numbers:
    average = sum(numbers) / len(numbers)
    print(f"Average: {average:.2f}")
else:
    print("No numbers found in file")
```

### Example 4: Simple Log File

```python
import datetime

# Add entry to log file
current_time = datetime.datetime.now()
log_entry = f"{current_time}: User logged in\n"

with open("activity.log", "a") as file:
    file.write(log_entry)

print("Activity logged successfully!")
```

## Handling Errors

Sometimes files don't exist or can't be opened. Use try-except to handle errors:

```python
try:
    with open("missing_file.txt", "r") as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("Sorry, the file doesn't exist!")
except PermissionError:
    print("You don't have permission to access this file!")
```

## Best Practices

1. **Always close files** - Use `with` statement to do this automatically
2. **Handle errors** - Files might not exist or be accessible
3. **Use appropriate modes** - Don't use "w" if you want to keep existing data
4. **Strip whitespace** - Use `.strip()` when reading lines to remove extra spaces/newlines
5. **Check if file exists** before trying to read it

## Quick Exercise

Try creating a simple program that:
1. Asks user for their favorite movies
2. Saves each movie to a file called "movies.txt"
3. Reads the file back and displays all movies

```python
# Solution
movies = []
while True:
    movie = input("Enter a movie (or 'done' to finish): ")
    if movie.lower() == 'done':
        break
    movies.append(movie)

# Save to file
with open("movies.txt", "w") as file:
    for movie in movies:
        file.write(movie + "\n")

# Read back from file
print("\nYour favorite movies:")
with open("movies.txt", "r") as file:
    for line in file:
        print(f"- {line.strip()}")
```

## Summary

File handling lets you:
- **Read** existing files with `open(filename, "r")`
- **Write** new files with `open(filename, "w")`
- **Append** to files with `open(filename, "a")`
- Use `with` statement for automatic file closing
- Handle errors with try-except blocks

Remember: files are like doors - open them, use them, and always close them when you're done!

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*
