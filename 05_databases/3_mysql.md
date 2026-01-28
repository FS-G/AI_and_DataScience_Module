# Beginner's MySQL Course - Complete Guide

## What is SQL?

SQL (Structured Query Language) is a language used to communicate with databases. Think of it as a way to ask questions to your data and get answers back.

## The Big Four: CRUD Operations

Before we dive deep, let's understand the four main things you can do with data:

1. **CREATE** (Insert) - Add new data
2. **READ** (Select) - Get data
3. **UPDATE** - Change existing data
4. **DELETE** - Remove data

Let's see quick examples:

```sql
-- READ: Get all students
SELECT * FROM students;

-- CREATE: Add a new student
INSERT INTO students (name, age) VALUES ('John', 20);

-- UPDATE: Change a student's age
UPDATE students SET age = 21 WHERE name = 'John';

-- DELETE: Remove a student
DELETE FROM students WHERE name = 'John';
```

Don't worry if this looks confusing - we'll explain everything step by step!

## Setting Up Our Practice Database

Let's create a simple school database that we'll use throughout this course:

### Database Structure Diagram

```mermaid
erDiagram
    STUDENTS {
        int student_id PK
        varchar name
        int age
        varchar email UK
        char grade
        date enrollment_date
    }
    
    COURSES {
        int course_id PK
        varchar course_name
        int credits
        varchar instructor
    }
    
    ENROLLMENTS {
        int enrollment_id PK
        int student_id FK
        int course_id FK
        decimal(5,2) grade
        date enrollment_date
    }
    
    STUDENTS ||--o{ ENROLLMENTS : "can have many"
    COURSES ||--o{ ENROLLMENTS : "can have many"
```

**Understanding the Diagram:**
- **STUDENTS** table stores student information
- **COURSES** table stores course information  
- **ENROLLMENTS** table connects students to courses (many-to-many relationship)
- **PK** = Primary Key, **FK** = Foreign Key, **UK** = Unique Key
- One student can enroll in many courses
- One course can have many students enrolled

### Create the Database

```sql
-- Create the database
CREATE DATABASE school_db;
USE school_db;

-- Create students table
CREATE TABLE students (
    student_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    age INT,
    email VARCHAR(100) UNIQUE,
    grade CHAR(1),
    enrollment_date DATE
);

-- Create courses table
CREATE TABLE courses (
    course_id INT PRIMARY KEY AUTO_INCREMENT,
    course_name VARCHAR(100) NOT NULL,
    credits INT,
    instructor VARCHAR(50)
);

-- Create enrollments table (connects students to courses)
CREATE TABLE enrollments (
    enrollment_id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT,
    course_id INT,
    grade DECIMAL(5,2),
    enrollment_date DATE,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

-- Insert sample students
INSERT INTO students (name, age, email, grade, enrollment_date) VALUES
('Alice Johnson', 20, 'alice@email.com', 'A', '2024-01-15'),
('Bob Smith', 21, 'bob@email.com', 'B', '2024-01-16'),
('Charlie Brown', 19, 'charlie@email.com', 'A', '2024-01-17'),
('Diana Lee', 22, 'diana@email.com', 'C', '2024-01-18'),
('Eve Wilson', 20, 'eve@email.com', 'B', '2024-01-19'),
('Frank Davis', 23, 'frank@email.com', 'A', '2024-01-20');

-- Insert sample courses
INSERT INTO courses (course_name, credits, instructor) VALUES
('Introduction to Programming', 3, 'Dr. Smith'),
('Database Systems', 4, 'Prof. Johnson'),
('Web Development', 3, 'Dr. Brown'),
('Data Structures', 4, 'Prof. Davis'),
('Computer Networks', 3, 'Dr. Wilson');

-- Insert sample enrollments
INSERT INTO enrollments (student_id, course_id, grade, enrollment_date) VALUES
(1, 1, 95.5, '2024-02-01'),
(1, 2, 87.0, '2024-02-01'),
(2, 1, 78.5, '2024-02-02'),
(2, 3, 92.0, '2024-02-02'),
(3, 2, 89.5, '2024-02-03'),
(3, 4, 94.0, '2024-02-03'),
(4, 1, 76.0, '2024-02-04'),
(4, 5, 82.5, '2024-02-04'),
(5, 3, 91.0, '2024-02-05'),
(6, 4, 88.0, '2024-02-06');
```

Great! Now we have a database with three tables and sample data to practice with.

## 1. Data Retrieval - The SELECT Statement

### Basic SELECT

The SELECT statement is like asking "Show me..."

```sql
-- Show all students
SELECT * FROM students;

-- Show only names and ages
SELECT name, age FROM students;

-- Show all courses
SELECT * FROM courses;
```

The `*` means "all columns". It's like saying "show me everything".

### Filtering with WHERE

WHERE is like adding conditions: "Show me... but only if..."

```sql
-- Students who are 20 years old
SELECT * FROM students WHERE age = 20;

-- Students with grade A
SELECT * FROM students WHERE grade = 'A';

-- Courses with 3 credits
SELECT * FROM courses WHERE credits = 3;
```

### Comparison Operators

```sql
-- Students older than 20
SELECT * FROM students WHERE age > 20;

-- Students 21 or younger
SELECT * FROM students WHERE age <= 21;

-- Students not in grade A
SELECT * FROM students WHERE grade != 'A';

-- Courses with less than 4 credits
SELECT * FROM courses WHERE credits < 4;
```

### Logical Operators

Combine multiple conditions:

```sql
-- Students who are 20 AND have grade A
SELECT * FROM students WHERE age = 20 AND grade = 'A';

-- Students who are either 19 OR 23 years old
SELECT * FROM students WHERE age = 19 OR age = 23;

-- Students who are NOT grade B
SELECT * FROM students WHERE NOT grade = 'B';

-- Students between 20-22 years old with grade A or B
SELECT * FROM students WHERE age BETWEEN 20 AND 22 AND (grade = 'A' OR grade = 'B');
```

### Sorting with ORDER BY

```sql
-- Students ordered by age (youngest first)
SELECT * FROM students ORDER BY age;

-- Students ordered by age (oldest first)
SELECT * FROM students ORDER BY age DESC;

-- Students ordered by grade, then by name
SELECT * FROM students ORDER BY grade, name;
```

### Limiting Results

```sql
-- Show only first 3 students
SELECT * FROM students LIMIT 3;

-- Show students 2-4 (skip first 1, then take 3)
SELECT * FROM students LIMIT 3 OFFSET 1;

-- Top 2 oldest students
SELECT * FROM students ORDER BY age DESC LIMIT 2;
```

### DISTINCT - Getting Unique Values

Sometimes you want to see unique values only, without duplicates:

```sql
-- Get unique grades (no duplicates)
SELECT DISTINCT grade FROM students;

-- Get unique ages
SELECT DISTINCT age FROM students;

-- Get unique instructors
SELECT DISTINCT instructor FROM courses;

-- Combine with other clauses
SELECT DISTINCT grade FROM students WHERE age > 20;

-- Count unique values
SELECT COUNT(DISTINCT grade) as unique_grades FROM students;
```

### Column Aliasing - Renaming Output

You can rename columns in your results using `AS`:

```sql
-- Give columns friendlier names
SELECT 
    name AS student_name,
    age AS student_age,
    email AS email_address
FROM students;

-- AS is optional (but recommended for clarity)
SELECT 
    name student_name,
    age student_age
FROM students;

-- Useful with calculations
SELECT 
    name,
    age,
    (2024 - age + 18) AS birth_year_estimate
FROM students;

-- With aggregates
SELECT 
    grade,
    COUNT(*) AS number_of_students,
    AVG(age) AS average_age
FROM students 
GROUP BY grade;
```

## 2. Advanced Filtering

### BETWEEN - Range Filtering

```sql
-- Students aged between 20 and 22
SELECT * FROM students WHERE age BETWEEN 20 AND 22;

-- Enrollments with grades between 80 and 90
SELECT * FROM enrollments WHERE grade BETWEEN 80 AND 90;
```

### IN - Multiple Value Matching

```sql
-- Students with grades A or B
SELECT * FROM students WHERE grade IN ('A', 'B');

-- Courses with 3 or 4 credits
SELECT * FROM courses WHERE credits IN (3, 4);

-- Students aged 19, 21, or 23
SELECT * FROM students WHERE age IN (19, 21, 23);
```

### LIKE - Pattern Matching

```sql
-- Students whose names start with 'A'
SELECT * FROM students WHERE name LIKE 'A%';

-- Students whose names end with 'son'
SELECT * FROM students WHERE name LIKE '%son';

-- Students with 'li' anywhere in their name
SELECT * FROM students WHERE name LIKE '%li%';

-- Students with exactly 8 characters in name (use _ for single character)
SELECT * FROM students WHERE name LIKE '________';

-- Email addresses from gmail
SELECT * FROM students WHERE email LIKE '%@gmail.com';
```

### Working with NULL Values

```sql
-- Find students with no email
SELECT * FROM students WHERE email IS NULL;

-- Find students who have an email
SELECT * FROM students WHERE email IS NOT NULL;

-- Find enrollments without grades
SELECT * FROM enrollments WHERE grade IS NULL;
```

## 3. Aggregation and Grouping

### Aggregate Functions

These functions calculate summary information:

```sql
-- Count total number of students
SELECT COUNT(*) FROM students;

-- Count students with email addresses
SELECT COUNT(email) FROM students;

-- Average age of students
SELECT AVG(age) FROM students;

-- Youngest and oldest student
SELECT MIN(age) as youngest, MAX(age) as oldest FROM students;

-- Total credits of all courses
SELECT SUM(credits) FROM courses;

-- Average grade in enrollments
SELECT AVG(grade) FROM enrollments;
```

### GROUP BY - Grouping Data

```sql
-- Count students by grade
SELECT grade, COUNT(*) as student_count 
FROM students 
GROUP BY grade;

-- Average age by grade
SELECT grade, AVG(age) as average_age 
FROM students 
GROUP BY grade;

-- Count enrollments per course
SELECT course_id, COUNT(*) as enrollment_count 
FROM enrollments 
GROUP BY course_id;

-- Credits per instructor
SELECT instructor, SUM(credits) as total_credits 
FROM courses 
GROUP BY instructor;
```

### HAVING - Filtering Groups

HAVING is like WHERE, but for groups:

```sql
-- Grades with more than 1 student
SELECT grade, COUNT(*) as student_count 
FROM students 
GROUP BY grade 
HAVING COUNT(*) > 1;

-- Courses with average grade above 85
SELECT course_id, AVG(grade) as avg_grade 
FROM enrollments 
GROUP BY course_id 
HAVING AVG(grade) > 85;

-- Instructors teaching more than 1 course
SELECT instructor, COUNT(*) as course_count 
FROM courses 
GROUP BY instructor 
HAVING COUNT(*) > 1;
```

## 4. Data Modification

### INSERT - Adding New Data

```sql
-- Add a single student
INSERT INTO students (name, age, email, grade, enrollment_date) 
VALUES ('John Doe', 19, 'john@email.com', 'B', '2024-02-15');

-- Add multiple students at once
INSERT INTO students (name, age, email, grade, enrollment_date) VALUES
('Mary Jane', 20, 'mary@email.com', 'A', '2024-02-16'),
('Peter Parker', 21, 'peter@email.com', 'B', '2024-02-17');

-- Add a course
INSERT INTO courses (course_name, credits, instructor) 
VALUES ('Machine Learning', 4, 'Dr. AI');
```

### UPDATE - Modifying Existing Data

**Safety Note:** MySQL has a safety feature that prevents updates without WHERE clauses. If you get a warning, run this first:

```sql
SET SQL_SAFE_UPDATES = 0;  -- Disable safe update mode
```

Now you can run update commands:

```sql
-- Update one student's grade
UPDATE students 
SET grade = 'A' 
WHERE name = 'John Doe';

-- Update multiple fields
UPDATE students 
SET age = 20, grade = 'A' 
WHERE name = 'John Doe';

-- Update based on condition
UPDATE enrollments 
SET grade = grade + 5 
WHERE grade < 80;

-- Update all courses by one instructor
UPDATE courses 
SET credits = 4 
WHERE instructor = 'Dr. Smith';
```

**Important:** Always use WHERE clause with UPDATE, or you'll update ALL rows!

### DELETE - Removing Data

```sql
-- Delete a specific student
DELETE FROM students WHERE name = 'John Doe';

-- Delete students with specific condition
DELETE FROM students WHERE age > 25;

-- Delete enrollments with low grades
DELETE FROM enrollments WHERE grade < 60;

-- Delete all data from table (be careful!)
DELETE FROM students;
```

### Table Modification - Changing Table Structure

Sometimes you need to change the table itself, not just the data:

#### Adding Columns
```sql
-- Add a phone column to students table
ALTER TABLE students ADD COLUMN phone VARCHAR(15);

-- Add multiple columns at once
ALTER TABLE students 
ADD COLUMN address TEXT,
ADD COLUMN city VARCHAR(50);

-- Add column with default value
ALTER TABLE students ADD COLUMN status VARCHAR(20) DEFAULT 'Active';
```

#### Removing Columns
```sql
-- Remove the phone column
ALTER TABLE students DROP COLUMN phone;

-- Remove multiple columns
ALTER TABLE students 
DROP COLUMN address,
DROP COLUMN city;
```

#### Modifying Existing Columns
```sql
-- Change the size of email field from 100 to 150 characters
ALTER TABLE students MODIFY COLUMN email VARCHAR(150);

-- Change data type
ALTER TABLE students MODIFY COLUMN age SMALLINT;

-- Rename a column
ALTER TABLE students CHANGE COLUMN grade letter_grade CHAR(1);
```

#### Real-World Example
```sql
-- Let's say we want to improve our students table:
-- 1. Add phone number
ALTER TABLE students ADD COLUMN phone VARCHAR(15);

-- 2. Make email field bigger
ALTER TABLE students MODIFY COLUMN email VARCHAR(150);

-- 3. Add a status column with default
ALTER TABLE students ADD COLUMN status VARCHAR(20) DEFAULT 'Active';

-- Check our changes
DESCRIBE students;
```

## 5. Table and Database Structure

### Data Types

Common MySQL data types:

```sql
-- Numeric types
INT              -- Whole numbers: 123, -456
DECIMAL(10,2)    -- Fixed decimal: 123.45 (10 total digits, 2 after decimal)
DECIMAL(5,2)     -- Grade scores: 95.50 (5 total digits, 2 after decimal)
FLOAT            -- Floating point numbers

-- Text types
VARCHAR(50)      -- Variable length text up to 50 characters
CHAR(2)          -- Fixed length text (exactly 2 characters)
TEXT             -- Long text

-- Date/Time types
DATE             -- Date: '2024-01-15'
DATETIME         -- Date and time: '2024-01-15 14:30:00'
TIME             -- Time only: '14:30:00'
```

### Creating Tables

```sql
-- Create a new table
CREATE TABLE teachers (
    teacher_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    subject VARCHAR(50),
    salary DECIMAL(10,2),
    hire_date DATE,
    email VARCHAR(100) UNIQUE
);
```

### Constraints - Rules for Data

Constraints are rules that ensure your data stays clean and consistent. Think of them as quality control for your database.

#### All SQL Constraints:

**1. NOT NULL** - Column cannot be empty
```sql
name VARCHAR(50) NOT NULL  -- Must have a value
```

**2. UNIQUE** - Value must be unique across all rows
```sql
email VARCHAR(100) UNIQUE  -- No duplicate emails
```

**3. PRIMARY KEY** - Unique identifier (combines NOT NULL + UNIQUE)
```sql
student_id INT PRIMARY KEY AUTO_INCREMENT
```

**4. FOREIGN KEY** - Links to another table
```sql
FOREIGN KEY (student_id) REFERENCES students(student_id)
```

**5. CHECK** - Custom validation rules
```sql
age INT CHECK (age >= 18 AND age <= 100)  -- Age must be 18-100
price DECIMAL(10,2) CHECK (price > 0)     -- Price must be positive
grade CHAR(1) CHECK (grade IN ('A','B','C','D','F'))
```

**6. DEFAULT** - Automatic value if none provided
```sql
status VARCHAR(20) DEFAULT 'Active'       -- Default to 'Active'
created_date DATE DEFAULT CURRENT_DATE    -- Default to today
```

**7. INDEX** - Improves search performance (not a constraint, but important)
```sql
CREATE INDEX idx_student_name ON students(name);
```



#### Why Use Constraints?
- **Data Quality**: Prevents bad data from entering your database
- **Consistency**: Ensures all data follows the same rules
- **Relationships**: Maintains connections between tables
- **Performance**: Some constraints (like PRIMARY KEY) improve speed

### Altering Tables

```sql
-- Add a new column
ALTER TABLE students ADD COLUMN phone VARCHAR(15);

-- Drop a column
ALTER TABLE students DROP COLUMN phone;

-- Modify a column
ALTER TABLE students MODIFY COLUMN email VARCHAR(150);

-- Add a constraint
ALTER TABLE students ADD CONSTRAINT unique_email UNIQUE (email);
```

### Primary and Foreign Keys

```sql
-- Primary Key: Unique identifier for each row
CREATE TABLE departments (
    dept_id INT PRIMARY KEY AUTO_INCREMENT,  -- Primary Key
    dept_name VARCHAR(50) NOT NULL
);

-- Foreign Key: Links to another table
CREATE TABLE employees (
    emp_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    dept_id INT,                            -- Foreign Key
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);
```

## 6. Joins - Combining Tables

Before we write queries, let's understand what joins do with simple table illustrations.

### Understanding Joins with Tables

Let's create two simple tables to demonstrate:

```sql
-- Create simple tables for demonstration
CREATE TABLE owners (
    owner_id INT PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE cars (
    car_id INT PRIMARY KEY,
    model VARCHAR(50),
    owner_id INT
);

-- Insert sample data
INSERT INTO owners VALUES 
(1, 'Alice'), (2, 'Bob'), (3, 'Charlie');

INSERT INTO cars VALUES 
(101, 'Toyota', 1), (102, 'Honda', 1), (103, 'BMW', 2);
```

Now our tables look like this:

**owners table:**
```
owner_id | name
---------|--------
1        | Alice
2        | Bob
3        | Charlie
```

**cars table:**
```
car_id | model  | owner_id
-------|--------|----------
101    | Toyota | 1
102    | Honda  | 1
103    | BMW    | 2
```

### Visual Join Examples

#### INNER JOIN
Shows only matching records from both tables:

**Result: owners INNER JOIN cars**
```
owner_id | name  | car_id | model
---------|-------|--------|--------
1        | Alice | 101    | Toyota
1        | Alice | 102    | Honda
2        | Bob   | 103    | BMW
```
*Notice: Charlie is missing because he has no cars*

#### LEFT JOIN
Shows ALL records from left table + matching from right:

**Result: owners LEFT JOIN cars**
```
owner_id | name    | car_id | model
---------|---------|--------|--------
1        | Alice   | 101    | Toyota
1        | Alice   | 102    | Honda
2        | Bob     | 103    | BMW
3        | Charlie | NULL   | NULL
```
*Notice: Charlie appears with NULL values for car info*

#### RIGHT JOIN
Shows ALL records from right table + matching from left:

**Result: owners RIGHT JOIN cars**
```
owner_id | name  | car_id | model
---------|-------|--------|--------
1        | Alice | 101    | Toyota
1        | Alice | 102    | Honda
2        | Bob   | 103    | BMW
```
*Same as INNER JOIN in this case (all cars have owners)*

### Now Let's Write the Actual Queries

### INNER JOIN

Shows only matching records from both tables:

```sql
-- Get owner names with their car models
SELECT owners.name, cars.model 
FROM owners 
INNER JOIN cars ON owners.owner_id = cars.owner_id;

-- Using our school database: students with their enrollment grades
SELECT students.name, enrollments.grade 
FROM students 
INNER JOIN enrollments ON students.student_id = enrollments.student_id;

-- Three table join: student names with course names they're enrolled in
SELECT students.name, courses.course_name, enrollments.grade
FROM students 
INNER JOIN enrollments ON students.student_id = enrollments.student_id
INNER JOIN courses ON enrollments.course_id = courses.course_id;
```

### LEFT JOIN

Shows all records from left table, matching ones from right:

```sql
-- All owners and their cars (including owners with no cars)
SELECT owners.name, cars.model 
FROM owners 
LEFT JOIN cars ON owners.owner_id = cars.owner_id;

-- All students and their grades (including students with no enrollments)
SELECT students.name, enrollments.grade 
FROM students 
LEFT JOIN enrollments ON students.student_id = enrollments.student_id;

-- All courses and enrollment count
SELECT courses.course_name, COUNT(enrollments.enrollment_id) as enrollment_count
FROM courses 
LEFT JOIN enrollments ON courses.course_id = enrollments.course_id
GROUP BY courses.course_id, courses.course_name;
```

### RIGHT JOIN

Shows all records from right table, matching ones from left:

```sql
-- All cars with their owners (including cars without owners - rare case)
SELECT owners.name, cars.model 
FROM owners 
RIGHT JOIN cars ON owners.owner_id = cars.owner_id;

-- All enrollments with student info
SELECT students.name, enrollments.grade 
FROM students 
RIGHT JOIN enrollments ON students.student_id = enrollments.student_id;
```

### CROSS JOIN

Creates all possible combinations (rarely used):

```sql
-- Every owner with every car model (not practical, just for learning)
SELECT owners.name, cars.model 
FROM owners 
CROSS JOIN cars;
```


### Using Table Aliases

Make queries shorter and clearer:

```sql
-- Using aliases (o for owners, c for cars)
SELECT o.name, c.model 
FROM owners o
INNER JOIN cars c ON o.owner_id = c.owner_id;

-- School database with aliases
SELECT s.name, c.course_name, e.grade 
FROM students s
INNER JOIN enrollments e ON s.student_id = e.student_id
INNER JOIN courses c ON e.course_id = c.course_id
WHERE e.grade > 85;
```

### Join Tips for Beginners

1. **Start with INNER JOIN** - it's the most common and easiest to understand
2. **Use LEFT JOIN** when you want "all from the first table"
3. **Always specify the join condition** with ON
4. **Use table aliases** to make queries readable
5. **Draw it out** - sketch the tables to visualize what you want

## 7. Subqueries

Subqueries are queries within queries.

### Basic Subqueries

```sql
-- Students older than average age
SELECT * FROM students 
WHERE age > (SELECT AVG(age) FROM students);

-- Students enrolled in 'Database Systems'
SELECT * FROM students 
WHERE student_id IN (
    SELECT student_id FROM enrollments 
    WHERE course_id = (
        SELECT course_id FROM courses 
        WHERE course_name = 'Database Systems'
    )
);
```

### Subqueries in SELECT

```sql
-- Show each student with their enrollment count
SELECT name, 
       (SELECT COUNT(*) FROM enrollments 
        WHERE enrollments.student_id = students.student_id) as enrollment_count
FROM students;
```

### Subqueries in FROM

```sql
-- Average grade by course (using subquery as table)
SELECT course_name, avg_grade
FROM (
    SELECT course_id, AVG(grade) as avg_grade 
    FROM enrollments 
    GROUP BY course_id
) as course_averages
JOIN courses ON course_averages.course_id = courses.course_id;
```

## 8. Practical Examples

Let's put it all together with some real-world examples:

### Example 1: Student Report Card
```sql
-- Complete report: student name, course, grade, instructor
SELECT 
    s.name as student_name,
    c.course_name,
    e.grade,
    c.instructor,
    c.credits
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
JOIN courses c ON e.course_id = c.course_id
ORDER BY s.name, c.course_name;
```

### Example 2: Course Performance Analysis
```sql
-- Which courses have the highest average grades?
SELECT 
    c.course_name,
    c.instructor,
    AVG(e.grade) as average_grade,
    COUNT(e.student_id) as student_count
FROM courses c
LEFT JOIN enrollments e ON c.course_id = e.course_id
GROUP BY c.course_id, c.course_name, c.instructor
HAVING COUNT(e.student_id) > 0
ORDER BY average_grade DESC;
```

### Example 3: Finding Top Students
```sql
-- Students with GPA above 90
SELECT 
    s.name,
    s.grade as letter_grade,
    AVG(e.grade) as gpa,
    COUNT(e.course_id) as courses_taken
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
GROUP BY s.student_id, s.name, s.grade
HAVING AVG(e.grade) > 90
ORDER BY gpa DESC;
```

## Quick Reference

### Common Commands
```sql
-- Database operations
CREATE DATABASE name;
USE database_name;
SHOW DATABASES;
DROP DATABASE name;

-- Table operations
SHOW TABLES;
DESCRIBE table_name;
DROP TABLE table_name;

-- Data operations
SELECT * FROM table_name;
INSERT INTO table_name (columns) VALUES (values);
UPDATE table_name SET column = value WHERE condition;
DELETE FROM table_name WHERE condition;
```

### Tips for Beginners

1. **Always test with SELECT first** - Before UPDATE or DELETE, use SELECT to see what rows you're affecting
2. **Use LIMIT when testing** - Add LIMIT 5 to see just a few rows while learning
3. **Comment your queries** - Use `--` for single line comments
4. **Format your queries** - Use proper indentation and line breaks for readability
5. **Start simple** - Begin with basic SELECT statements, then add complexity
6. **Practice regularly** - SQL is best learned by doing

## Next Steps

Once you're comfortable with these basics:

1. Learn about **indexes** for better performance
2. Explore **views** for reusable queries
3. Study **stored procedures** and **functions**
4. Learn about **transactions** and **locks**
5. Understand **database design** and **normalization**

Remember: SQL is a powerful language, but it's also logical and consistent. Practice with real data, start simple, and gradually build complexity. The key is to understand what you're asking the database to do in plain English first, then translate that into SQL.

Happy querying!