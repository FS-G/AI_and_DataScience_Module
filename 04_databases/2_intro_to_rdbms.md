# Database Fundamentals & ERD Design - Complete Lecture

## What is Data? What is a Database?

### Data
• Information in raw or organized form
• Can be numbers, text, images, videos, audio files
• Examples: your name, age, purchase history, photos
• Becomes valuable when organized and analyzed

### Database
• Organized collection of structured information stored electronically
• Like a digital filing cabinet with rules
• Allows efficient storage, retrieval, and management of data
• Examples: your phone's contact list, Netflix's movie catalog, bank records

---

## Database Management Systems (DBMS)

### What is DBMS?
• Software that manages databases
• Acts as interface between users and database
• Handles storage, retrieval, security, backup

### Key Functions
• **Data Storage**: Organizes data efficiently on disk
• **Data Retrieval**: Finds and returns requested information quickly
• **Security**: Controls who can access what data
• **Backup & Recovery**: Protects against data loss
• **Concurrency**: Allows multiple users simultaneously

### Popular DBMS Examples
• MySQL, PostgreSQL, Oracle, SQL Server, MongoDB

---

## Database Paradigms: An Overview

### What are Database Paradigms?
• Different approaches to organizing and storing data
• Each suited for different types of applications
• Two main categories: Relational and Non-Relational

### Why Different Paradigms?
• No single solution fits all problems
• Different data types need different storage methods
• Performance requirements vary by application
• Scalability needs differ

---

## Relational Databases (SQL / Structured)

### Key Characteristics
• Data stored in tables with rows and columns
• Uses Structured Query Language (SQL)
• Follows ACID properties (Atomicity, Consistency, Isolation, Durability)
• Enforces data integrity through constraints

### When to Use
• Well-defined data structure
• Need complex queries and transactions
• Data consistency is critical
• Examples: banking systems, inventory management

### Advantages
• Data integrity and consistency
• Complex queries possible
• Mature technology with wide support
• Standardized SQL language

---

## Non-Relational Databases (NoSQL)

### Key Characteristics
• Flexible data models (not just tables)
• Horizontally scalable
• Often sacrifice consistency for performance/availability
• Various types for different use cases

### When to Use
• Rapidly changing data structures
• Need to scale across many servers
• Handle large volumes of unstructured data
• Examples: social media feeds, IoT sensor data

### Types of NoSQL Databases

#### Key-Value Stores
• **Structure**: Simplest NoSQL model - data stored as key-value pairs
• **Like**: Giant hash table or dictionary
• **Characteristics**: 
  - Key: Unique identifier (like a locker number)
  - Value: Any type of data (text, numbers, files)
  - Very fast lookups by key, limited querying capabilities
• **Use Cases**: Session management, user preferences, shopping carts, caching
• **Examples**: Redis, Amazon DynamoDB, Riak, MongoDB (can also function as key-value store)

#### Document Databases
• **Structure**: Store data as documents (usually JSON-like)
• **Characteristics**: 
  - Documents can contain nested structures
  - No fixed schema required
  - Documents are self-contained, schema flexibility
• **Use Cases**: Content management systems, product catalogs, user profiles, blog posts
• **Examples**: MongoDB, CouchDB, Amazon DocumentDB

#### Graph Databases
• **Structure**: Data stored as nodes (entities) and edges (relationships)
• **Characteristics**: 
  - Focus on relationships between data points
  - Optimized for traversing relationships
  - Queries follow paths through the graph
• **Use Cases**: Social networks (friends, followers), recommendation engines, fraud detection, knowledge graphs
• **Examples**: Neo4j, Amazon Neptune, ArangoDB

#### Vector Databases
• **Structure**: Store data as high-dimensional vectors
• **Characteristics**: 
  - Each vector represents features of an item
  - Uses mathematical distance to find similar items
  - Excellent for AI and machine learning applications
• **Use Cases**: Image recognition and search, natural language processing, recommendation systems, AI-powered applications
• **Examples**: Pinecone, Weaviate, Chroma

---

## Course Focus: The Relational Model

### Why Focus on Relational?
• Foundation for understanding databases
• Most widely used in business applications
• Strong theoretical foundation
• Transferable skills to other paradigms

### Core Concepts We'll Master
• Entity-Relationship Diagrams (ERDs)
• Table design and normalization
• Keys and constraints
• SQL querying and manipulation

---

## Designing the Blueprint: Entity-Relationship Diagrams (ERDs)

### What is an ERD?
• Visual map of your database before you build it
• Shows tables and how they connect
• Like a blueprint for a house

### Why Use ERDs?
• Plan before coding (saves time later)
• Easy to explain to others
• Spot problems early

### ERD Has 3 Parts
• **Entities**: Future database tables
• **Attributes**: Future table columns
• **Relationships**: How tables connect

---

## Entities, Attributes, and Relationships

### Entities = Future Tables
• Things you want tables for
• Draw as rectangles
• Use nouns: Customer, Product, Order
• Think: "What do I need a table for?"

### Attributes = Future Columns
• Information stored in each table
• Draw as ovals connected to rectangles
• Examples: Customer Name, Product Price, Order Date
• Think: "What info goes in this table?"

### Types of Attributes
• **Simple**: One piece of info (Age, Price)
• **Composite**: Can split up (Full Name = First + Last)
• **Derived**: Calculate from others (Age from Birth Date)
• **Multi-valued**: Multiple entries (Phone Numbers)

### Relationships = How Tables Connect
• Show connections between tables
• Draw as diamonds
• Use verbs: Customer BUYS Product, Student TAKES Course
• Think: "How do these tables relate?"

---

## Cardinality: One-to-One, One-to-Many, Many-to-Many

### What is Cardinality?
• Describes how many instances of one entity relate to instances of another
• Critical for proper database design
• Affects how we structure tables

### One-to-One (1:1)
• One instance relates to exactly one instance
• Relatively rare in practice
• **Example**: Person HAS ONE Social Security Number
• **Example**: Country HAS ONE Capital City

### One-to-Many (1:M)
• One instance relates to many instances
• Most common relationship type
• **Example**: Customer PLACES many Orders
• **Example**: Department HAS many Employees

### Many-to-Many (M:N)
• Many instances relate to many instances
• Requires junction table in implementation
• **Example**: Students ENROLL IN multiple Courses, Courses HAVE multiple Students
• **Example**: Authors WRITE multiple Books, Books HAVE multiple Authors

### How to Identify Cardinality
• Ask: "How many X can relate to one Y?"
• Ask: "How many Y can relate to one X?"
• Consider real-world business rules

---

## The Building Blocks: Tables, Rows, and Columns

### Tables (Relations)
• Two-dimensional structure storing data
• Represents an entity from ERD
• Has a unique name
• Collection of related data

### Rows (Tuples/Records)
• Horizontal data entries
• Each row represents one instance of the entity
• Example: One row = one customer, one product, one order

### Columns (Attributes/Fields)
• Vertical data categories
• Each column represents one property
• Has a specific data type
• Example: Name column, Price column, Date column

### Table Rules
• Each table must have a primary key
• Column names must be unique within table
• Each cell contains single value (atomic)
• Order of rows and columns doesn't matter

---

## Understanding Data Types

### Why Data Types Matter?
• Determines how data is stored and processed
• Affects storage space and performance
• Enables data validation
• Supports appropriate operations

### Numeric Types
• **INTEGER**: Whole numbers (-2, 0, 42)
• **DECIMAL/NUMERIC**: Precise decimal numbers (19.99, 0.05)
• **FLOAT/REAL**: Approximate decimal numbers (scientific calculations)
• **BOOLEAN**: True/False values

### Text Types
• **CHAR(n)**: Fixed-length text (exactly n characters)
• **VARCHAR(n)**: Variable-length text (up to n characters)
• **TEXT**: Long text without specific limit

### Date/Time Types
• **DATE**: Calendar dates (2024-12-25)
• **TIME**: Time of day (14:30:00)
• **TIMESTAMP**: Date and time together
• **INTERVAL**: Time duration

### Choosing the Right Type
• Consider the data you'll store
• Think about storage efficiency
• Plan for future needs
• Follow business requirements

---

## Keys: Primary, Foreign, and Unique

### Primary Key
• Uniquely identifies each row in a table
• Cannot be NULL
• Cannot be duplicated
• Should be stable (doesn't change)
• Examples: Customer ID, Product SKU, Order Number

**Example: Students Table**
```
StudentID | FirstName | LastName | Email
----------|-----------|----------|------------------
1001      | John      | Smith    | john@email.com
1002      | Sarah     | Johnson  | sarah@email.com
1003      | Mike      | Brown    | mike@email.com
```
*StudentID is the Primary Key*

### Foreign Key
• Links to primary key in another table
• Creates relationships between tables
• Can be NULL (unless specified otherwise)
• Must match existing primary key value
• Examples: Customer ID in Orders table

**Example: Enrollments Table**
```
EnrollmentID | StudentID | CourseCode | Grade
-------------|-----------|------------|-------
501          | 1001      | CS101      | A
502          | 1002      | CS101      | B+
503          | 1001      | MATH201    | A-
```
*StudentID is a Foreign Key linking to Students table*

### Unique Key
• Ensures column values are unique
• Can be NULL (but only one NULL typically)
• Table can have multiple unique keys
• Examples: Email address, Social Security Number

**In Students table above:**
• Email is a Unique Key (no two students can have same email)
• StudentID is Primary Key AND unique
• FirstName is NOT unique (multiple Johns allowed)

### Composite Keys
• Key made up of multiple columns
• All columns together must be unique
• Useful when single column isn't sufficient
• Example: Course ID + Student ID for enrollment

**Example: Course Schedule Table**
```
CourseCode | Semester | Room | Instructor
-----------|----------|------|------------
CS101      | Fall2024 | A101 | Dr. Smith
CS101      | Spring2025| B205 | Dr. Jones
MATH201    | Fall2024 | C301 | Dr. Wilson
```
*Composite Key: CourseCode + Semester (same course can run multiple semesters)*

---


## E-Commerce Database Design Class Activity

### Business Scenario
You're designing a database for "ShopEasy" - an online retail platform where:
• Customers register and place orders
• Each order contains multiple products
• Products belong to categories
• Customers can review products
• We track inventory and shipping

### Your Task (15 minutes)
Working in groups, identify:
1. **Entities**: What "things" do we need to store data about?
2. **Attributes**: What properties does each entity have?
3. **Relationships**: How do entities connect?
4. **Cardinalities**: One-to-one, one-to-many, or many-to-many?

### Think About
• What information do customers provide?
• What details do we need for products?
• How do we track orders and their contents?
• What about categories, reviews, and inventory?

---

## E-Commerce ERD Solution

### Identified Entities
• **Customer**: People who shop on our platform
• **Product**: Items available for purchase
• **Category**: Product classifications
• **Order**: Purchase transactions
• **OrderItem**: Individual products within an order
• **Review**: Customer feedback on products
• **Inventory**: Stock tracking for products

### Entity Attributes

**Customer**
• CustomerID (Primary Key)
• FirstName, LastName
• Email (Unique)
• Phone
• Address, City, State, ZipCode
• DateJoined
• IsActive

**Product**
• ProductID (Primary Key)
• ProductName
• Description
• Price
• CategoryID (Foreign Key)
• Brand
• Weight
• Dimensions
• IsActive

**Category**
• CategoryID (Primary Key)
• CategoryName
• Description
• ParentCategoryID (Foreign Key - for subcategories)

**Order**
• OrderID (Primary Key)
• CustomerID (Foreign Key)
• OrderDate
• TotalAmount
• Status (Pending, Shipped, Delivered, Cancelled)
• ShippingAddress
• PaymentMethod

**OrderItem**
• OrderItemID (Primary Key)
• OrderID (Foreign Key)
• ProductID (Foreign Key)
• Quantity
• UnitPrice
• Subtotal

**Review**
• ReviewID (Primary Key)
• ProductID (Foreign Key)
• CustomerID (Foreign Key)
• Rating (1-5)
• ReviewText
• ReviewDate

**Inventory**
• InventoryID (Primary Key)
• ProductID (Foreign Key)
• QuantityInStock
• ReorderLevel
• LastRestocked

### Key Relationships & Cardinalities

**Customer → Order** (1:M)
• One customer can place many orders
• Each order belongs to exactly one customer

**Order → OrderItem** (1:M)
• One order contains many order items
• Each order item belongs to exactly one order

**Product → OrderItem** (1:M)
• One product can appear in many order items
• Each order item refers to exactly one product

**Category → Product** (1:M)
• One category contains many products
• Each product belongs to exactly one category

**Category → Category** (1:M) - Self-referencing
• One category can have many subcategories
• Each subcategory has one parent category

**Customer → Review** (1:M)
• One customer can write many reviews
• Each review is written by exactly one customer

**Product → Review** (1:M)
• One product can have many reviews
• Each review is about exactly one product

**Product → Inventory** (1:1)
• One product has exactly one inventory record
• Each inventory record tracks exactly one product

### Design Decisions Explained

**Why OrderItem Entity?**
• Resolves many-to-many between Order and Product
• Stores quantity and price at time of purchase
• Allows same product multiple times in one order

**Why Separate Inventory?**
• Isolates stock management from product info
• Allows different access permissions
• Supports inventory-specific operations

**Self-Referencing Category**
• Supports category hierarchies (Electronics → Computers → Laptops)
• Flexible structure for organizing products
• Enables navigation breadcrumbs

**Price in Both Product and OrderItem**
• Product.Price = current price
• OrderItem.UnitPrice = price when purchased
• Protects against price changes affecting historical orders

below is the schematic view (mermaid script) for the current design.

```
erDiagram
    CUSTOMER {
        int CustomerID PK
        string FirstName
        string LastName
        string Email UK
        string Phone
        string Address
        string City
        string State
        string ZipCode
        date DateJoined
        boolean IsActive
    }

    PRODUCT {
        int ProductID PK
        string ProductName
        string Description
        decimal Price
        int CategoryID FK
        string Brand
        string Weight
        string Dimensions
        boolean IsActive
    }

    CATEGORY {
        int CategoryID PK
        string CategoryName
        string Description
        int ParentCategoryID FK
    }

    ORDER {
        int OrderID PK
        int CustomerID FK
        date OrderDate
        decimal TotalAmount
        string Status
        string ShippingAddress
        string PaymentMethod
    }

    ORDERITEM {
        int OrderItemID PK
        int OrderID FK
        int ProductID FK
        int Quantity
        decimal UnitPrice
        decimal Subtotal
    }

    REVIEW {
        int ReviewID PK
        int ProductID FK
        int CustomerID FK
        int Rating
        string ReviewText
        date ReviewDate
    }

    INVENTORY {
        int InventoryID PK
        int ProductID FK
        int QuantityInStock
        int ReorderLevel
        date LastRestocked
    }

    CUSTOMER ||--o{ ORDER : places
    ORDER ||--o{ ORDERITEM : contains
    PRODUCT ||--o{ ORDERITEM : appears_in
    CATEGORY ||--o{ PRODUCT : contains
    CATEGORY ||--o{ CATEGORY : subcategory_of
    CUSTOMER ||--o{ REVIEW : writes
    PRODUCT ||--o{ REVIEW : receives
    PRODUCT ||--|| INVENTORY : tracked_by
```


---

## DB Normalization vs De-Normalization

Database normalization organizes data to eliminate redundancy and store related data in separate tables, enhancing efficiency, maintainability, and data accuracy. For example, a customer’s address is stored once in a Customer table and linked to orders, reducing duplication. Denormalization, conversely, intentionally combines tables or duplicates data to improve read performance, such as embedding a customer’s address in each order record for faster reporting, though it may require extra effort to maintain consistency. In summary, normalization prioritizes accuracy and structure, while denormalization optimizes for speed and convenience.


---

## Survey of Common Relational Database Systems

### Factors to Consider
• **Cost**: Open source vs commercial licensing
• **Performance**: Speed for your specific workload
• **Scalability**: Growth capacity
• **Features**: Built-in functionality
• **Support**: Community vs enterprise support
• **Skills**: Team expertise and learning curve

---

## MySQL: Pros & Cons

### Pros
• **Free & Open Source**: No licensing costs
• **Easy to Learn**: Simple setup and administration
• **Wide Adoption**: Large community, lots of resources
• **Good Performance**: Fast for read-heavy applications
• **Web-Friendly**: Excellent for web applications (LAMP stack)
• **Cloud Support**: Available on all major cloud platforms

### Cons
• **Limited Features**: Fewer advanced features than competitors
• **ACID Compliance**: Some storage engines don't fully support ACID
• **Complex Queries**: Can struggle with very complex operations
• **Enterprise Features**: Advanced features require paid version
• **Backup Tools**: Built-in backup options are limited

### Best For
• Web applications and content management
• Small to medium-sized businesses
• Development and testing environments
• Applications prioritizing simplicity

---

## PostgreSQL: Pros & Cons

### Pros
• **Full ACID Compliance**: Robust transaction support
• **Advanced Features**: JSON, arrays, custom data types
• **Extensibility**: Custom functions, operators, and data types
• **Standards Compliance**: Follows SQL standards closely
• **No Licensing Costs**: Completely free
• **Great Performance**: Excellent for complex queries

### Cons
• **Memory Usage**: Can consume more RAM than MySQL
• **Learning Curve**: More complex for beginners
• **Configuration**: Requires more tuning for optimal performance
• **Smaller Community**: Less widespread than MySQL (but growing)

### Best For
• Complex applications with advanced data needs
• Data warehousing and analytics
• Applications requiring custom data types
• Organizations prioritizing standards compliance

---

## Microsoft SQL Server: Pros & Cons

### Pros
• **Enterprise Features**: Advanced security, analytics, reporting
• **Microsoft Integration**: Seamless with Microsoft ecosystem
• **Great Tools**: Excellent management and development tools
• **Performance**: Very good for large enterprise applications
• **Support**: Professional Microsoft support available
• **Business Intelligence**: Built-in BI and reporting tools

### Cons
• **Cost**: Expensive licensing, especially for enterprise features
• **Windows-Centric**: Primarily designed for Windows environments
• **Vendor Lock-in**: Tied to Microsoft technology stack
• **Resource Usage**: Can be memory and CPU intensive

### Best For
• Large enterprises using Microsoft technologies
• Applications requiring advanced BI features
• Organizations with dedicated database administrators
• Windows-based infrastructure

---

## Oracle Database: Pros & Cons

### Pros
• **Enterprise Scale**: Handles largest, most complex databases
• **Advanced Features**: Cutting-edge database technology
• **High Availability**: Excellent disaster recovery and clustering
• **Security**: Industry-leading security features
• **Performance**: Optimized for high-transaction environments
• **Maturity**: Decades of development and refinement

### Cons
• **Very Expensive**: Highest licensing and maintenance costs
• **Complexity**: Requires specialized Oracle DBAs
• **Over-Engineering**: May be overkill for smaller applications
• **Vendor Lock-in**: Proprietary features make migration difficult
• **Learning Curve**: Steep learning curve for administrators

### Best For
• Large enterprises with complex data needs
• Mission-critical applications requiring maximum uptime
• Organizations with dedicated Oracle expertise
• Applications handling massive transaction volumes

---

## Choosing the Right Database

### Decision Framework
1. **Budget**: What can you afford for licensing and support?
2. **Scale**: How much data and how many users?
3. **Complexity**: How complex are your data relationships and queries?
4. **Skills**: What expertise does your team have?
5. **Integration**: What other systems need to connect?
6. **Growth**: How will your needs change over time?

### General Recommendations
• **Starting Out**: PostgreSQL or MySQL
• **Enterprise with Microsoft Stack**: SQL Server
• **Maximum Scale & Features**: Oracle
• **Web Applications**: MySQL or PostgreSQL
• **Complex Analytics**: PostgreSQL or SQL Server
• **Budget Constrained**: PostgreSQL or MySQL

### Remember
• You can often migrate between systems later
• Start with what your team knows
• Consider managed cloud options
• Focus on getting started rather than perfect choice

