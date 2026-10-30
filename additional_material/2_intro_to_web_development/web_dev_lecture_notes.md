**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Web Development Lecture Notes


### The Big Picture
- **What happens when you visit a website?**
  - You type URL → Browser sends request → Server responds → Page appears
  - Like calling a restaurant: you ask for menu, they send it back
- **Static vs Dynamic websites**
  - **Static**: Same content for everyone (like a brochure)
  - **Dynamic**: Changes based on user (like your Facebook feed)


### The Three Main Components
- **Frontend**: What users see and interact with
  - The restaurant's dining area and menu
  - Buttons, forms, colors, animations
- **Backend**: The invisible logic and processing
  - The restaurant's kitchen where food is prepared
  - Business rules, calculations, data processing
- **Database**: Where information is stored
  - The restaurant's pantry and inventory
  - User accounts, posts, product catalogs
- **Simple analogy**: Restaurant (frontend = dining area, backend = kitchen, database = pantry)

### How They Communicate
- **The request-response cycle** explained simply
  - User clicks button → Frontend sends request → Backend processes → Database queried → Response sent back
- **Real example**: Logging into Facebook
  - Enter username/password → Sent to Facebook servers → Checked against database → Welcome page returned

---


### What is Frontend Development?
- **User interface and user experience**
  - Everything the user sees and touches
  - Making websites beautiful and easy to use
- **The three building blocks**: HTML, CSS, JavaScript
  - Work together like a team
  - Each has a specific job
- **Responsive design** - working on all devices
  - Same website works on phone, tablet, computer
  - Adapts to different screen sizes

### HTML - The Structure
- **Like the skeleton of a webpage**
  - Defines what content exists
  - Headings, paragraphs, links, images, forms
- **Semantic meaning** - why structure matters
  - Tells browsers and search engines what things are
  - Helps people with disabilities use websites
- **Basic elements**: `<h1>`, `<p>`, `<a>`, `<img>`, `<form>`

### CSS - The Styling
- **Making things look good**
  - Colors, fonts, spacing, layouts
  - Like decorating a house after it's built
- **Layout systems**: Flexbox, Grid
  - How to arrange elements on the page
- **Responsive design**: Media queries
  - Different styles for different screen sizes
- **CSS frameworks** - pre-built styles
  - Bootstrap, Tailwind - like interior design templates

### JavaScript - The Behavior
- **Making pages interactive**
  - Responding to clicks, form submissions
  - Changing content without page reload
- **Event handling**: What happens when user acts
  - Click, scroll, type, hover
- **DOM manipulation**: Changing page content dynamically
  - Add, remove, or modify elements
- **Modern frameworks** overview
  - **React**: Component-based development
  - **Vue**: Progressive framework
  - **Angular**: Full application framework

---



---


### What is Backend Development?
- **The server-side logic**
  - Code that runs on servers, not in browsers
  - Handles business rules and data processing
- **Popular languages**: Python, JavaScript, Java, PHP
  - **Python**: Easy to learn, powerful libraries
  - **JavaScript (Node.js)**: Same language as frontend
  - **Java**: Enterprise applications
  - **PHP**: Web-focused, powers WordPress

### APIs - How Systems Talk
- **What is an API?**
  - **Application Programming Interface**
  - Like a waiter between kitchen and customer
  - Defines how software components communicate
- **REST APIs** explained simply
  - **RE**presentational **S**tate **T**ransfer
  - Uses standard HTTP methods
  - Stateless - each request is independent
- **JSON** - the data format
  - **JavaScript Object Notation**
  - Human-readable way to structure data
  - `{"name": "John", "age": 30}`
- **HTTP methods**: 
  - **GET**: Retrieve data (like asking for menu)
  - **POST**: Create new data (like placing order)
  - **PUT**: Update existing data (like changing order)
  - **DELETE**: Remove data (like canceling order)
- **Beyond HTTP**: Other ways systems communicate
  - **WebSockets**: Two-way real-time communication
    - **Use case**: WhatsApp messages, live gaming, stock prices
    - **Why**: HTTP asks once, gets answer. WebSockets keep connection open
  - **GraphQL**: Ask for exactly what data you need
    - **Use case**: Mobile apps that need less data, complex dashboards
    - **Why**: Instead of getting full user profile, just ask for name and email
    - **Example**: Traditional API gives you everything (name, email, address, phone, settings, friends list). GraphQL lets you say "I only want name and email" - saves bandwidth and loading time
  - **gRPC**: Fast communication between services
    - **Use case**: Microservices talking to each other, high-traffic apps
    - **Why**: Faster than HTTP, works well for internal systems

### Server Infrastructure
- **Web servers and hosting**
  - **Cloud platforms**: AWS, Google Cloud, Azure
- **Scalability basics**
  - Handling more users and data over time
  - Load balancing, caching strategies

### Security Fundamentals
- **Authentication**: Who are you?
  - Login systems, passwords, tokens
- **Authorization**: What can you do?
  - Permissions and access control
- **HTTPS**: Keeping data safe in transit
  - Encryption between browser and server
- **Common vulnerabilities**: SQL injection, XSS, CSRF



### Why We Need Databases
- **Storing and organizing data**
  - Structured way to keep information
  - Better than files for complex data
- **Persistence** - remembering information
  - Data survives server restarts
  - Multiple users can access same data
- **CRUD operations**: Create, Read, Update, Delete
  - Basic things you do with data

### SQL Databases
- **Tables, rows, and columns**
  - Like Excel spreadsheets that talk to each other
  - **Table**: Collection of related data
  - **Row**: Individual record
  - **Column**: Specific data field
- **Relationships between data**
  - **One-to-many**: One user has many posts
  - **Many-to-many**: Students enrolled in courses
- **Popular databases**: 
  - **MySQL**: Most common, easy to learn
  - **PostgreSQL**: More features, enterprise-grade
  - **SQLite**: Lightweight, perfect for small apps

### NoSQL Databases
- **When SQL isn't the best fit**
  - Flexible data structures
  - Rapid development and changes
- **Document databases**: MongoDB
  - Store data as JSON-like documents
  - Good for content management
- **Key-value stores**: Redis
  - Simple storage, very fast
  - Good for caching and sessions
- **When to use which type**
  - **SQL**: Structured data, complex relationships
  - **NoSQL**: Flexible schemas, rapid scaling

---


### A holistic view of a web application

A web application is a set of cooperating parts. Each part has a clear responsibility, and the boundaries between them make the system easier to understand and change.

| Part | Main responsibility | Example in this lesson |
|---|---|---|
| Browser / frontend | Presents information, collects user input, and sends requests | Searchable book catalog and loan status badges |
| API | Defines the available operations and the shape of requests and responses | `GET /api/books`, `POST /api/loans` |
| Backend | Validates requests, applies rules, coordinates work, and returns results | Checks that a book can be borrowed |
| Database | Stores durable, related records and answers queries | PostgreSQL tables for books, students, staff, and loans |

The browser does not connect directly to the database. It talks to the backend through an API. The backend is the trusted middle layer: it checks input and permissions, applies business rules, and asks the database to read or change records. The database enforces data relationships and stores committed changes so that they remain after the application restarts.

### API and communication essentials

An **API (Application Programming Interface)** is a defined way for one part of a system to ask another part to do something. In a web application, it is a contract between the frontend and backend:

- **Available operations:** URLs and HTTP methods, such as `GET /api/books`.
- **Expected input:** required fields and their format, such as `book_id` and `member_id` for a loan.
- **Expected output:** response data and status codes for success or errors.
- **Separation of responsibilities:** the frontend can use the API without knowing how the backend stores data internally.

**HTTP** is the request-response protocol commonly used for communication between a browser and a web server. A typical exchange follows this sequence:

1. **The browser prepares a request** for an API operation.
2. **The method** says what the browser wants to do, such as `GET` to read or `POST` to create.
3. **The URL** identifies the resource or operation, such as `/api/books`.
4. **Headers** carry request metadata, such as `Content-Type: application/json` to describe the body format.
5. **The body (optional)** carries data being sent, such as a book and member ID when creating a loan.
6. **The server processes the request** and sends back a response.
7. **The response** includes a status code, headers, and often a body containing data or an error message.

**HTTP is stateless by default:** each request should include the information the server needs to handle it. The server does not automatically remember earlier requests.

**Methods describe intent.** Common conventions are:

| Method | Typical purpose | Library example | Typical success status |
|---|---|---|---|
| `GET` | Read information; should not change stored data | `GET /api/books?search=history` searches books | `200 OK` (including an empty list if there are no matches) |
| `POST` | Create a resource or ask the server to perform an operation | `POST /api/loans` creates a loan | `201 Created` when a resource was created; sometimes `200 OK` for an action that returns a result |
| `PUT` | Replace the resource in full. Send the complete new representation, including unchanged fields. | `PUT /api/books/12` replaces the book's editable details. | `200 OK` with the updated resource, or `204 No Content` without a response body; `201 Created` if it creates a new resource. |
| `PATCH` | Change only selected fields. Send just the fields being changed; other fields stay as they are. | `PATCH /api/loans/27` could set only `returned_on`. | `200 OK` with the updated resource, or `204 No Content` without a response body. |
| `DELETE` | Remove a resource | `DELETE /api/books/12` removes a book if allowed | `204 No Content` is common; `200 OK` may be used if the response includes a confirmation or representation |

These are conventions, not automatic behavior. The API designer chooses and documents the response:

- **`GET`:** normally returns `200 OK` when successful, including when the result is an empty list.
- **`POST`:** normally returns `201 Created` when it creates a resource; the response may include the new loan and its URL.
- **`PUT`:** replaces the full resource representation. It often returns `200 OK` with the updated record or `204 No Content` without one. It may return `201 Created` if it creates a resource.
- **`PATCH`:** changes selected fields. It often returns `200 OK` with the updated record or `204 No Content` without one.
- **`DELETE`:** commonly returns `204 No Content`, because there is no response body; an API may return `200 OK` if it includes a confirmation or representation.

**Status codes tell the caller what happened.** The first digit gives a broad category:

- **`2xx` — Success:** the server handled the request.
- **`4xx` — Request problem:** the request cannot be fulfilled as sent.
- **`5xx` — Server problem:** the server failed while handling an apparently valid request.

Common codes in this project include:

- **`200 OK`:** read or update succeeded.
- **`201 Created`:** a new resource, such as a loan, was created.
- **`204 No Content`:** the request succeeded, but the response has no body.
- **`400 Bad Request`:** the input is invalid or violates a business rule.
- **`401 Unauthorized`:** authentication is missing or invalid.
- **`403 Forbidden`:** the caller is identified but lacks permission.
- **`404 Not Found`:** the requested record does not exist.
- **`409 Conflict`:** the request conflicts with current data, such as borrowing a book that is already on loan.
- **`422 Unprocessable Content`:** the request data fails validation; FastAPI commonly uses this when fields do not match the expected schema.
- **`500 Internal Server Error`:** an unexpected server failure occurred.

The frontend should **check the response status** and show a useful message for errors. It should not assume that every response contains valid success data.

**JSON (JavaScript Object Notation)** is a common format for request and response bodies:

- **Request example:** `{"book_id": 12, "member_id": 4}` asks to create a loan.
- **Content type:** the browser labels the body as `application/json`; FastAPI parses and validates it.
- **Response example:** `{"id": 27, "book_id": 12, "member_id": 4, "due_on": "2026-11-14", "status": "on_loan"}` gives the frontend the created loan details.
- **Dates:** are commonly sent in a documented string format such as ISO `YYYY-MM-DD`, then parsed and handled as dates by the backend.
- **Meaning:** JSON carries the data; the API contract defines what the operation and fields mean.

**Communication path:** follow the data through the application in order:

1. **Browser JavaScript** calls `fetch()` with a URL, method, headers, and optional body.
2. **HTTP request** travels over the network to FastAPI.
3. **FastAPI** validates the request and routes it to backend logic.
4. **Backend logic** applies the business rules and may query PostgreSQL.
5. **HTTP response** returns a status code and, often, JSON data.
6. **Browser JavaScript** reads the response and updates the page.



**Validation happens at more than one layer:**

- **Browser checks** catch simple mistakes early and improve usability.
- **FastAPI checks** that incoming data has the expected types and required fields.
- **Application rules** check whether an operation makes sense, such as whether the book is available.
- **PostgreSQL constraints** protect stored data even if an application path has a bug.

Client-side checks are never a substitute for server-side validation.

**CORS (Cross-Origin Resource Sharing)** is a browser security rule for requests across different origins:

- **An origin** is the combination of scheme, host, and port. In `http://localhost:5500`, the scheme is `http`, the host is `localhost`, and the port is `5500`.
- **Example:** `http://localhost:5500` and `http://localhost:8000` differ because their ports are different.
- **Local setup:** configure FastAPI to allow the specific frontend origin.
- **Important distinction:** CORS is not authentication; it controls which browser pages may read responses. Do not allow every origin in a deployed application without a reason.

**Other communication patterns** suit other needs:

- **WebSockets:** keep a two-way connection open for live updates such as chat or notifications.
- **GraphQL:** exposes a schema that lets clients request selected fields.
- **gRPC:** is often used for efficient service-to-service calls.

A small browser application can begin with ordinary HTTP and JSON because it is easy to inspect, test, and explain.

### Frontend responsibilities

- **Where it runs:** In the user's browser.
- **HTML:** Gives the page structure and meaning.
- **CSS:** Controls presentation and responsive layout.
- **JavaScript:** Responds to events, calls the API with `fetch`, and renders returned data.
- **User experience:** Provides search, loading feedback, empty states, and clear error messages.
- **Validation:** Browser checks help users, but the backend must repeat important checks because browser checks can be bypassed.

### Backend responsibilities

- **Where it runs:** As a server process, separate from the user's browser.
- **FastAPI:** Maps HTTP routes to Python functions, validates JSON with request/response schemas, and provides interactive API documentation.
- **Route handlers:** Receive validated input, call focused application logic, and translate results into HTTP responses. Keep them small.
- **Business rules:** Belong in application logic, not in the browser or in large route handlers.
- **Database access:** Goes through a database/repository layer so the rest of the application does not need connection details.
- **Authentication:** Identifies who the user is.
- **Authorization:** Checks what the identified user is allowed to do.
- **Security basics:** Use HTTPS, keep secrets out of source code, validate inputs, use parameterized SQL, and avoid exposing internal error details to clients.

### Database responsibilities and relationships

- **PostgreSQL:** A relational database that stores structured, durable application data.
- **Tables, columns, and rows:** A table holds one kind of record; columns describe its fields; rows are individual records.
- **Primary key:** Uniquely identifies a row, such as `books.id`.
- **Foreign key:** Connects related rows and helps prevent references to records that do not exist.
- **Constraints:** Rules such as `NOT NULL` and `UNIQUE` help protect data quality.
- **Transactions:** Group database operations so they either succeed together or roll back together.

For the library, the central relationship is:

```text
books (one) ──── (many) loans (many) ──── (one) students
```

- **Loan relationships:** Each loan references one book, one student (the library member), and the staff member who issued it. A book, student, or staff member can appear in many historical loans.
- **Example columns:** `books.id`, `books.title`, `students.id`, `students.name`, `staff.id`, `staff.name`, and `loans.id`, `loans.book_id`, `loans.student_id`, `loans.staff_id`, `loans.checked_out_on`, `loans.due_on`, and nullable `loans.returned_on`.
- **Dates:** Store dates as date/time values, not formatted display strings.
- **Active loan:** Has no return date.
- **Overdue loan:** Is active and has a due date earlier than today. Query this from the dates instead of saving a status that may become stale tomorrow.

- **SQL databases:** Fit data with clear structure and relationships, joins, and reliable constraints.
- **NoSQL databases:** Document stores and key-value stores can fit other data shapes and access patterns. They are alternatives for particular needs, not automatic upgrades to relational databases.

### How the parts are operated

- **Frontend and backend:** Run as separate processes during development.
- **Database:** PostgreSQL runs in a Docker container.
- **Configuration:** Database host, port, name, username, and password belong in the backend `.env` file, not in frontend JavaScript or committed source code.
- **Connection:** The backend reads `.env` settings to connect to PostgreSQL.
- **Persistence:** Docker provides a repeatable database environment; a named volume keeps database files when the container is stopped or recreated.

- **In production:** The parts may run on separate machines or managed services.
- **Frontend hosting:** A web server or hosting platform serves frontend files.
- **Backend hosting:** An application server runs FastAPI.
- **Database hosting:** A managed PostgreSQL service stores data.
- **Operations:** Logs, backups, monitoring, HTTPS, access control, and deployment configuration help keep the system running.
- **Architecture:** The same frontend, API/backend, and database roles remain as infrastructure grows.

---



### Project goal

Build a small web app where a library keeps track of books, student members, staff, and loans. Students can borrow books through staff and return them, and the system flags any loan that is past its due date. It is a familiar concept, so students can focus on how the web application works instead of learning unfamiliar domain rules.

This project introduces:

- **Foreign keys**, since loans link a book and student to a staff member
- **Date handling**, for due dates and late calculations
- **Useful queries**, like finding overdue loans

On the frontend, students build a searchable catalog and show a status badge (`On loan`, `Returned`, or `Overdue`) for each record. The goal is to follow the journey of data through the browser, API, backend, and database. Students should be able to put the provided code into the named files, start the services, and observe that flow; this is not an exercise in inventing code or learning advanced framework techniques.

### Follow a library action through the system

As the first example, suppose a librarian clicks **Borrow** for a book:

1. HTML provides the button and form; CSS makes the page readable on different screen sizes.
2. Browser JavaScript handles the click and sends an HTTP `POST` request, usually with JSON containing the member and book IDs.
3. FastAPI matches the URL and method to an endpoint, validates the request shape, and calls the library logic.
4. Backend logic checks that the member and book exist and that the book is available.
5. The backend sends SQL through a database driver. PostgreSQL inserts a loan row with a checkout date and due date, within a transaction.
6. The backend returns an HTTP status and JSON response. The browser reads it and updates the catalog to show the new status.

The other actions follow the same path. Searching sends a `GET` request with a search term; the backend queries PostgreSQL and returns matching books as JSON. Returning a book sends a `PATCH` request; the backend records the return date and returns the updated loan. In each case, the user action travels through the frontend, API, backend logic, and database, then the response travels back to the browser.

### Suggested project layout

Use small modules grouped by responsibility. Each module contains one main class named after its module. Keep one public operation per class where practical; helper methods are private and start with double underscores. For example, `library.py` contains `Library`, whose public method could be `borrow_book`; internal checks can be `__book_is_available` and `__make_due_date`.

```text
library_app/
├── docker-compose.yml
├── .gitignore
├── backend/
│   ├── .env                 # Local secrets and connection settings; do not commit
│   ├── .env.example         # Safe template with placeholder values
│   ├── requirements.txt
│   └── app/
│       ├── main.py          # Creates the FastAPI app and includes routers
│       ├── core/
│       │   └── config.py    # Reads settings from backend/.env
│       ├── db/
│       │   ├── connection.py # Creates database engine/session
│       │   └── base.py      # Shared ORM metadata/base, if using an ORM
│       ├── models/          # Database table models: Book, Member, Loan
│       ├── schemas/         # API request and response shapes
│       ├── repositories/    # Focused database reads/writes
│       ├── services/
│       │   ├── library.py   # Library class; borrowing/return rules
│       │   └── overdue.py   # Overdue class; overdue-loan query/use case
│       └── api/
│           └── routes/
│               ├── books.py
│               ├── students.py
│               └── loans.py
└── frontend/
    ├── index.html
    ├── styles.css
    └── app.js
```

The exact module names can vary. Preserve the simple boundaries: routes handle HTTP, schemas describe data exchanged over the API, services hold use-case rules, repositories handle persistence, models describe database structure, and `core`/`db` handle configuration and connections. If teaching a very small first version, several directories may initially contain only one file; the structure is there to show each responsibility clearly.

### Database ERD: plan the tables before starting PostgreSQL

Before starting the database, review this **entity-relationship diagram (ERD)**. It shows the four tables the activity creates, their important columns, and how a loan connects the other records. In the diagram, `PK` means primary key and `FK` means foreign key.

```mermaid
erDiagram
    BOOKS ||--o{ LOANS : "is checked out in"
    STUDENTS ||--o{ LOANS : "borrows"
    STAFF ||--o{ LOANS : "issues"

    BOOKS {
        int id PK
        string title
        string author
        string isbn UK
        int year_published
    }
    STUDENTS {
        int id PK
        string name
        string email UK
        string phone
    }
    STAFF {
        int id PK
        string name
        string email UK
        string role
    }
    LOANS {
        int id PK
        int book_id FK
        int student_id FK
        int staff_id FK
        date checked_out_on
        date due_on
        date returned_on "nullable until returned"
    }
```

- **One book can have many loan records over time.** A loan points to exactly one book.
- **One student can have many loan records.** A loan points to exactly one student who borrowed the book.
- **One staff member can issue many loans.** Each loan records the staff member who processed it.
- **`returned_on` is nullable:** an empty value means the loan is still active. The app derives `on_loan`, `overdue`, or `returned` from the dates rather than storing a status that can become out of date.
- **Unique values:** `books.isbn`, `students.email`, and `staff.email` must not be duplicated.
- **Delete rule:** foreign keys restrict deletion of books, students, or staff that are referenced by loan history.

The SQL in `backend/app/db/setup.py` implements this ERD. Looking at the diagram first helps students understand why the loans table has three foreign keys and why PostgreSQL can prevent accidental deletion of referenced records.

### Start PostgreSQL with Docker

Install and start Docker Desktop first. From the project root, save this as `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16
    container_name: library-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: library_db
      POSTGRES_USER: library_user
      POSTGRES_PASSWORD: library_password
    ports:
      - "5432:5432"
    volumes:
      - library_data:/var/lib/postgresql/data

volumes:
  library_data:
```

Start the database from the project root:

```bash
docker compose up -d
```

Check that it is running with `docker compose ps`. Stop the container with `docker compose down`; the named volume retains the database. To remove the database data as well, `docker compose down -v` deletes that volume, so use it only when you intend to reset the demo.

Create `backend/.env` and have backend settings read this file:

```env
DATABASE_URL=postgresql+psycopg://library_user:library_password@localhost:5432/library_db
FRONTEND_ORIGIN=http://localhost:5500
```

These values are local teaching credentials only. Add `.env` to `.gitignore`; commit `.env.example` with placeholder values instead. When the backend itself runs inside Docker, the database host in the URL becomes the Compose service name `postgres`, rather than `localhost`.

### API map for the project

Keep the API understandable. This guided version provides CRUD operations for students, books, and staff, plus the loan workflow:

| Method and path | Purpose | Example result |
|---|---|---|
| `GET /api/students?search=amina` | List or search students | Student records |
| `POST /api/students` | Create a student | New student (`201`) |
| `PUT /api/students/{id}` | Replace a student's details | Updated student |
| `DELETE /api/students/{id}` | Delete a student with no loan history | `204`; conflict if loans reference the student |
| `GET /api/books?search=history` | Search catalog and show current status | Books with `available`, `on_loan`, or `overdue` status |
| `POST /api/books`, `PUT /api/books/{id}`, `DELETE /api/books/{id}` | Create, replace, or delete a book | Book record; deletion is blocked if loan history exists |
| `GET /api/staff`, `POST /api/staff`, `PUT /api/staff/{id}`, `DELETE /api/staff/{id}` | Manage staff records | Staff records |
| `GET /api/loans` | List loans with book, student, and staff names | Loan history and current state |
| `POST /api/loans` | Check out an available book to a student by a staff member | New loan and due date (`201`) |
| `PATCH /api/loans/{loan_id}/return` | Return an active loan | Updated loan with return date |
| `GET /api/loans/overdue` | Find active loans whose due dates have passed | Loan, book, student, staff, and due date |

For example, a borrow request might send `{"book_id": 12, "student_id": 4, "staff_id": 2}` as JSON. FastAPI validates that these are integers, the `Library` use case checks availability, and PostgreSQL enforces the foreign keys. The response might include `{"id": 27, "due_on": "2026-11-14", "status": "on_loan"}`. In a complete implementation, use consistent timezone/date rules and define whether a loan due today becomes overdue at the start or end of the due date.

### What students should observe

1. Starting Docker starts PostgreSQL; it does not start FastAPI or the browser page.
2. Starting FastAPI loads backend settings from `.env`, connects to PostgreSQL, and makes API routes available. FastAPI's interactive documentation is at `http://localhost:8000/docs`.
3. The static frontend is served from `http://localhost:5500` by a simple development server (for example, `python -m http.server 5500` from the `frontend` directory). Serving the files over HTTP avoids browser `file://` restrictions.
4. Managing students, books, and staff uses `GET`, `POST`, `PUT`, and `DELETE`; the browser updates its tables after the API confirms each change.
5. Searching makes the browser send a `GET`; the backend asks PostgreSQL for matches and sends JSON back; JavaScript renders the catalog and status badges.
6. Borrowing makes the browser send a `POST`; the backend checks the rules and writes a loan; the response gives the frontend the due date to display.
7. Returning makes the browser send a `PATCH`; the database records the return date. The status then changes to returned.
8. Passing the due date makes an unreturned loan overdue. The backend's date comparison and overdue query support that badge and the overdue list.

The important outcome is a mental model of the complete cycle: a user action becomes an HTTP request; an API route validates and dispatches it; backend logic applies rules; PostgreSQL reads or stores related data; the backend returns a response; the frontend renders the result. Each file should be simple enough for students to run and trace without needing to design the architecture themselves.

### Guided coding activity: build and run the library app

In this activity, students copy the code into the named files and run the application. Read each file's short responsibility, then trace student, book, and staff creation, a search, a borrow, and a return from the browser to the database and back. The code uses simple SQL through `psycopg` so the database operations remain visible; it keeps a small module per task and one main class per module. To keep the example manageable, the CRUD operations are grouped in the `Library` service while the HTTP routes stay in FastAPI.

#### 1. Create the project files

Create the folders and files below. Add `backend/.env` using the values from the Docker setup above. Add this `.gitignore` at the project root:

```gitignore
backend/.env
backend/.venv/
__pycache__/
```

In `backend/requirements.txt`, list the backend packages:

```txt
fastapi
uvicorn[standard]
psycopg[binary]
python-dotenv
```

From the `backend` folder, create and activate a virtual environment, then install the packages:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. Backend configuration and database

`backend/app/core/config.py` reads connection settings from `.env`:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class Config:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5500")
        if not self.database_url:
            raise RuntimeError("Set DATABASE_URL in backend/.env")
```

`backend/app/db/connection.py` opens short-lived PostgreSQL connections:

```python
import psycopg
from app.core.config import Config


class Connection:
    def __init__(self):
        self.__database_url = Config().database_url

    def open(self):
        return psycopg.connect(self.__database_url)
```

`backend/app/db/setup.py` creates the tables and a few starter records. Loans reference books, students, and the staff member who issued them. Foreign keys use `ON DELETE RESTRICT` so records with loan history cannot be deleted accidentally; `returned_on` remains empty until a book is returned.

```python
from app.db.connection import Connection


class Setup:
    def __init__(self):
        self.__connection = Connection()

    def create_tables_and_sample_data(self):
        with self.__connection.open() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    author TEXT NOT NULL,
                    isbn TEXT NOT NULL UNIQUE,
                    year_published INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    phone TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS staff (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL DEFAULT 'Librarian'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS loans (
                    id SERIAL PRIMARY KEY,
                    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE RESTRICT,
                    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE RESTRICT,
                    staff_id INTEGER NOT NULL REFERENCES staff(id) ON DELETE RESTRICT,
                    checked_out_on DATE NOT NULL,
                    due_on DATE NOT NULL,
                    returned_on DATE
                )
            """)
            conn.execute("""
                INSERT INTO books (title, author, isbn, year_published)
                SELECT v.title, v.author, v.isbn, v.year_published
                FROM (VALUES
                    ('The Hobbit', 'J.R.R. Tolkien', '9780547928227', 1937),
                    ('Pride and Prejudice', 'Jane Austen', '9780141439518', 1813),
                    ('A Brief History of Time', 'Stephen Hawking', '9780553380163', 1988)
                ) AS v(title, author, isbn, year_published)
                WHERE NOT EXISTS (SELECT 1 FROM books)
            """)
            conn.execute("""
                INSERT INTO students (name, email, phone)
                SELECT v.name, v.email, v.phone
                FROM (VALUES
                    ('Amina Khan', 'amina@example.com', '0300-1111111'),
                    ('Omar Ali', 'omar@example.com', '0300-2222222')
                ) AS v(name, email, phone)
                WHERE NOT EXISTS (SELECT 1 FROM students)
            """)
            conn.execute("""
                INSERT INTO staff (name, email, role)
                SELECT v.name, v.email, v.role
                FROM (VALUES ('Sara Ahmed', 'sara@example.com', 'Librarian')) AS v(name, email, role)
                WHERE NOT EXISTS (SELECT 1 FROM staff)
            """)
```

`backend/app/db/setup.py` runs schema setup once when the API starts. The connection context commits successful changes and closes the connection. The starter data provides a student and staff member to make borrowing available immediately.

#### 3. Library operations

`backend/app/services/library.py` contains the main library operations. The public methods perform the use cases; double-underscore helpers contain internal queries and checks.

```python
from datetime import date, timedelta
from app.db.connection import Connection


class Library:
    def __init__(self):
        self.__connection = Connection()

    def list_students(self, search=""):
        return self.__list_records("students", search)

    def create_student(self, data):
        return self.__create_person("students", data, ("name", "email", "phone"))

    def update_student(self, student_id, data):
        return self.__update_person("students", student_id, data, ("name", "email", "phone"))

    def delete_student(self, student_id):
        return self.__delete_record("students", student_id)

    def list_staff(self):
        return self.__list_records("staff")

    def create_staff(self, data):
        return self.__create_person("staff", data, ("name", "email", "role"))

    def update_staff(self, staff_id, data):
        return self.__update_person("staff", staff_id, data, ("name", "email", "role"))

    def delete_staff(self, staff_id):
        return self.__delete_record("staff", staff_id)

    def list_books(self, search=""):
        with self.__connection.open() as conn:
            rows = conn.execute("""
                SELECT b.id, b.title, b.author, b.isbn, b.year_published,
                       CASE
                           WHEN l.returned_on IS NULL AND l.due_on < CURRENT_DATE THEN 'overdue'
                           WHEN l.returned_on IS NULL THEN 'on_loan'
                           ELSE 'available'
                       END AS status
                FROM books b
                LEFT JOIN loans l ON l.book_id = b.id AND l.returned_on IS NULL
                WHERE b.title ILIKE %s OR b.author ILIKE %s OR b.isbn ILIKE %s
                ORDER BY b.title
            """, (f"%{search}%", f"%{search}%", f"%{search}%")).fetchall()
        return [dict(row) for row in rows]

    def create_book(self, data):
        with self.__connection.open() as conn:
            row = conn.execute("""
                INSERT INTO books (title, author, isbn, year_published)
                VALUES (%s, %s, %s, %s)
                RETURNING id, title, author, isbn, year_published
            """, (data.title, data.author, data.isbn, data.year_published)).fetchone()
        return dict(row)

    def update_book(self, book_id, data):
        with self.__connection.open() as conn:
            row = conn.execute("""
                UPDATE books SET title = %s, author = %s, isbn = %s, year_published = %s
                WHERE id = %s
                RETURNING id, title, author, isbn, year_published
            """, (data.title, data.author, data.isbn, data.year_published, book_id)).fetchone()
        return dict(row) if row else None

    def delete_book(self, book_id):
        return self.__delete_record("books", book_id)

    def list_staff_for_select(self):
        with self.__connection.open() as conn:
            rows = conn.execute("SELECT id, name FROM staff ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def list_loans(self):
        with self.__connection.open() as conn:
            rows = conn.execute("""
                SELECT l.id, b.title AS book_title, s.name AS student_name,
                       f.name AS staff_name, l.checked_out_on, l.due_on, l.returned_on,
                       CASE WHEN l.returned_on IS NOT NULL THEN 'returned'
                            WHEN l.due_on < CURRENT_DATE THEN 'overdue'
                            ELSE 'on_loan' END AS status
                FROM loans l
                JOIN books b ON b.id = l.book_id
                JOIN students s ON s.id = l.student_id
                JOIN staff f ON f.id = l.staff_id
                ORDER BY l.checked_out_on DESC, l.id DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def borrow_book(self, book_id, student_id, staff_id):
        with self.__connection.open() as conn:
            if not self.__book_is_available(conn, book_id):
                return None
            today = date.today()
            row = conn.execute("""
                INSERT INTO loans (book_id, student_id, staff_id, checked_out_on, due_on)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, book_id, student_id, staff_id, checked_out_on, due_on, returned_on
            """, (book_id, student_id, staff_id, today, today + timedelta(days=14))).fetchone()
        return dict(row)

    def return_book(self, loan_id):
        with self.__connection.open() as conn:
            row = conn.execute("""
                UPDATE loans SET returned_on = CURRENT_DATE
                WHERE id = %s AND returned_on IS NULL
                RETURNING id, book_id, student_id, staff_id, checked_out_on, due_on, returned_on
            """, (loan_id,)).fetchone()
        return dict(row) if row else None

    def list_overdue(self):
        with self.__connection.open() as conn:
            rows = conn.execute("""
                SELECT l.id AS loan_id, b.title, s.name AS student_name,
                       f.name AS staff_name, l.due_on
                FROM loans l
                JOIN books b ON b.id = l.book_id
                JOIN students s ON s.id = l.student_id
                JOIN staff f ON f.id = l.staff_id
                WHERE l.returned_on IS NULL AND l.due_on < CURRENT_DATE
                ORDER BY l.due_on
            """).fetchall()
        return [dict(row) for row in rows]

    def __book_is_available(self, conn, book_id):
        row = conn.execute("""
            SELECT NOT EXISTS (
                SELECT 1 FROM loans WHERE book_id = %s AND returned_on IS NULL
            ) AS available
        """, (book_id,)).fetchone()
        return row["available"]

    def __list_records(self, table, search=""):
        with self.__connection.open() as conn:
            if table == "students":
                rows = conn.execute("""
                    SELECT id, name, email, phone FROM students
                    WHERE name ILIKE %s OR email ILIKE %s ORDER BY name
                """, (f"%{search}%", f"%{search}%")).fetchall()
            else:
                rows = conn.execute("SELECT id, name, email, role FROM staff ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def __create_person(self, table, data, fields):
        columns = ", ".join(fields)
        placeholders = ", ".join(["%s"] * len(fields))
        values = tuple(getattr(data, field) for field in fields)
        with self.__connection.open() as conn:
            row = conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING *", values
            ).fetchone()
        return dict(row)

    def __update_person(self, table, record_id, data, fields):
        assignments = ", ".join(f"{field} = %s" for field in fields)
        values = tuple(getattr(data, field) for field in fields) + (record_id,)
        with self.__connection.open() as conn:
            row = conn.execute(
                f"UPDATE {table} SET {assignments} WHERE id = %s RETURNING *", values
            ).fetchone()
        return dict(row) if row else None

    def __delete_record(self, table, record_id):
        with self.__connection.open() as conn:
            row = conn.execute(
                f"DELETE FROM {table} WHERE id = %s RETURNING id", (record_id,)
            ).fetchone()
        return bool(row)
```

In `backend/app/db/connection.py`, set rows to dictionary-like objects so the code above can return JSON-friendly dictionaries. Update its imports and `open` method as follows:

```python
import psycopg
from psycopg.rows import dict_row
from app.core.config import Config


class Connection:
    def __init__(self):
        self.__database_url = Config().database_url

    def open(self):
        return psycopg.connect(self.__database_url, row_factory=dict_row)
```

#### 4. FastAPI routes

`backend/app/main.py` sets up the tables, allows the local frontend origin, and exposes CRUD endpoints for students, books, and staff, as well as loan operations. FastAPI uses Pydantic models to validate incoming JSON.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.core.config import Config
from app.db.setup import Setup
from app.services.library import Library

config = Config()
library = Library()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Setup().create_tables_and_sample_data()
    yield


app = FastAPI(title="Library API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.frontend_origin],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


class LoanRequest(BaseModel):
    book_id: int
    student_id: int
    staff_id: int


class StudentInput(BaseModel):
    name: str
    email: str
    phone: str = ""


class StaffInput(BaseModel):
    name: str
    email: str
    role: str = "Librarian"


class BookInput(BaseModel):
    title: str
    author: str
    isbn: str
    year_published: int


def handle_delete(table, record_id):
    try:
        deleted = getattr(library, f"delete_{table}")(record_id)
    except Exception as error:
        raise HTTPException(status_code=409, detail="Record is in use or conflicts with existing data") from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Record was not found")
    return None


@app.get("/api/students")
def get_students(search: str = ""):
    return library.list_students(search)


@app.post("/api/students", status_code=201)
def create_student(request: StudentInput):
    try:
        return library.create_student(request)
    except Exception as error:
        raise HTTPException(status_code=409, detail="Student email already exists") from error


@app.put("/api/students/{student_id}")
def update_student(student_id: int, request: StudentInput):
    result = library.update_student(student_id, request)
    if result is None:
        raise HTTPException(status_code=404, detail="Student was not found")
    return result


@app.delete("/api/students/{student_id}", status_code=204)
def delete_student(student_id: int):
    return handle_delete("student", student_id)


@app.get("/api/staff")
def get_staff():
    return library.list_staff()


@app.post("/api/staff", status_code=201)
def create_staff(request: StaffInput):
    try:
        return library.create_staff(request)
    except Exception as error:
        raise HTTPException(status_code=409, detail="Staff email already exists") from error


@app.put("/api/staff/{staff_id}")
def update_staff(staff_id: int, request: StaffInput):
    result = library.update_staff(staff_id, request)
    if result is None:
        raise HTTPException(status_code=404, detail="Staff member was not found")
    return result


@app.delete("/api/staff/{staff_id}", status_code=204)
def delete_staff(staff_id: int):
    return handle_delete("staff", staff_id)


@app.get("/api/books")
def get_books(search: str = ""):
    return library.list_books(search)


@app.post("/api/books", status_code=201)
def create_book(request: BookInput):
    try:
        return library.create_book(request)
    except Exception as error:
        raise HTTPException(status_code=409, detail="ISBN already exists") from error


@app.put("/api/books/{book_id}")
def update_book(book_id: int, request: BookInput):
    result = library.update_book(book_id, request)
    if result is None:
        raise HTTPException(status_code=404, detail="Book was not found")
    return result


@app.delete("/api/books/{book_id}", status_code=204)
def delete_book(book_id: int):
    return handle_delete("book", book_id)


@app.get("/api/loans")
def get_loans():
    return library.list_loans()


@app.post("/api/loans", status_code=201)
def borrow_book(request: LoanRequest):
    try:
        loan = library.borrow_book(request.book_id, request.student_id, request.staff_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail="Book or member was not found") from error
    if loan is None:
        raise HTTPException(status_code=409, detail="This book is already on loan")
    return loan


@app.patch("/api/loans/{loan_id}/return")
def return_book(loan_id: int):
    loan = library.return_book(loan_id)
    if loan is None:
        raise HTTPException(status_code=404, detail="Active loan was not found")
    return loan


@app.get("/api/loans/overdue")
def get_overdue_loans():
    return library.list_overdue()
```

#### 5. Frontend files

`frontend/index.html` provides a small admin-style dashboard. The tabs separate the catalog, student records, staff records, and loan desk so each task is easy to find:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Library Desk</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header><div><p class="eyebrow">LIBRARY MANAGEMENT</p><h1>Library Desk</h1></div><span class="live">● System ready</span></header>
  <main>
    <nav class="tabs" aria-label="Library sections">
      <button class="tab active" data-view="catalog">Catalog</button>
      <button class="tab" data-view="students">Students</button>
      <button class="tab" data-view="staff">Staff</button>
      <button class="tab" data-view="loans">Loans</button>
    </nav>
    <p id="message" role="status"></p>
    <section id="catalog-view" class="view">
      <div class="section-heading"><div><h2>Book catalog</h2><p>Search titles and manage the collection.</p></div><button id="add-book" class="primary">Add book</button></div>
      <input id="search" type="search" placeholder="Search title, author, or ISBN">
      <div id="books" class="records"></div>
    </section>
    <section id="students-view" class="view hidden">
      <div class="section-heading"><div><h2>Students</h2><p>Manage library members.</p></div><button id="add-student" class="primary">Add student</button></div>
      <input id="student-search" type="search" placeholder="Search students">
      <div id="students" class="records"></div>
    </section>
    <section id="staff-view" class="view hidden">
      <div class="section-heading"><div><h2>Staff</h2><p>Manage the people who issue and return books.</p></div><button id="add-staff" class="primary">Add staff</button></div>
      <div id="staff" class="records"></div>
    </section>
    <section id="loans-view" class="view hidden">
      <div class="section-heading"><div><h2>Loans</h2><p>Track checkouts, returns, and overdue books.</p></div></div>
      <form id="loan-form" class="loan-form">
        <label>Book<select id="loan-book" required></select></label>
        <label>Student<select id="loan-student" required></select></label>
        <label>Issued by<select id="loan-staff" required></select></label>
        <button class="primary">Check out book</button>
      </form>
      <h3>Loan history</h3><div id="loans" class="records"></div>
      <h3>Overdue</h3><div id="overdue" class="records"></div>
    </section>
    <dialog id="record-dialog">
      <form id="record-form" method="dialog">
        <div class="section-heading"><h2 id="dialog-title">Add record</h2><button type="button" id="close-dialog" class="icon">×</button></div>
        <div id="dialog-fields"></div><div class="dialog-actions"><button type="button" id="cancel-dialog">Cancel</button><button class="primary">Save record</button></div>
      </form>
    </dialog>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

`frontend/styles.css` gives the dashboard a clear layout, readable tables, responsive forms, and distinct loan-state badges:

```css
* { box-sizing: border-box; }
body { font: 15px Arial, sans-serif; margin: 0; background: #f3f6fa; color: #203047; }
header { display: flex; justify-content: space-between; align-items: center; padding: 1.2rem max(1.25rem, calc((100% - 1050px)/2)); background: #173b67; color: white; }
h1, h2, h3, p { margin-top: 0; } h1 { margin-bottom: 0; } h2 { margin-bottom: .35rem; } h3 { margin-top: 1.5rem; }
.eyebrow { margin: 0 0 .3rem; font-size: .7rem; letter-spacing: .14em; opacity: .75; } .live { color: #c7f5db; font-size: .85rem; }
main { max-width: 1050px; margin: 1.5rem auto; padding: 0 1rem; }
.tabs { display: flex; gap: .35rem; border-bottom: 1px solid #d9e1eb; margin-bottom: 1.3rem; }
button, input, select { padding: .65rem .75rem; font: inherit; border: 1px solid #ccd6e2; border-radius: 7px; background: white; color: inherit; }
button { cursor: pointer; } button:hover { background: #eef3f8; } .tab { border: 0; background: transparent; border-radius: 7px 7px 0 0; } .tab.active { color: #145cb5; box-shadow: inset 0 -3px #2876d2; font-weight: bold; }
.primary { background: #2168b5; color: white; border-color: #2168b5; } .primary:hover { background: #174f8d; color: white; }
.section-heading { display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin: .5rem 0 1.1rem; } .section-heading p { color: #69788d; margin: 0; }
.view { background: white; border: 1px solid #e0e6ee; border-radius: 12px; padding: 1.3rem; box-shadow: 0 5px 20px #1b35520b; } .hidden { display: none; }
#search, #student-search { width: min(100%, 420px); margin-bottom: 1rem; }
.record { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 1rem .25rem; border-top: 1px solid #e8edf3; }
.record-main { flex: 1; } .record-title { font-weight: bold; margin-bottom: .3rem; } .muted { color: #68778b; font-size: .9rem; }
.actions { display: flex; gap: .4rem; } .actions button { padding: .4rem .6rem; font-size: .85rem; }
.badge { display: inline-block; border-radius: 1rem; padding: .3rem .65rem; background: #e7eef5; font-size: .8rem; text-transform: capitalize; white-space: nowrap; }
.badge.overdue { background: #ffe0df; color: #8b1e18; } .badge.on_loan { background: #fff0c2; color: #694d00; } .badge.available, .badge.returned { background: #dff5e8; color: #17633b; }
.loan-form { display: flex; flex-wrap: wrap; align-items: end; gap: .8rem; padding: 1rem; background: #f5f8fb; border-radius: 9px; } .loan-form label { display: grid; gap: .4rem; flex: 1; min-width: 170px; }
dialog { width: min(460px, calc(100% - 2rem)); border: 0; border-radius: 12px; padding: 1.4rem; box-shadow: 0 20px 80px #132b4880; } dialog::backdrop { background: #12253b88; } #dialog-fields label { display: grid; gap: .4rem; margin-bottom: .9rem; } #dialog-fields input { width: 100%; }
.dialog-actions { display: flex; justify-content: end; gap: .5rem; margin-top: 1.2rem; } .icon { border: 0; font-size: 1.4rem; padding: .25rem .6rem; }
#message { min-height: 1.3rem; color: #235f9f; }
@media (max-width: 650px) { header { padding: 1rem; } main { margin-top: .8rem; padding: 0 .6rem; } .view { padding: .9rem; } .record { align-items: flex-start; flex-wrap: wrap; } .actions { width: 100%; } .tabs { overflow-x: auto; } }
```

`frontend/app.js` calls the API, builds the CRUD forms, and renders the catalog and loan tables. The browser does not connect to PostgreSQL. This small helper-driven file keeps the example readable while all data changes still travel through HTTP:

```javascript
const API = "http://localhost:8000";
const message = document.querySelector("#message");
const dialog = document.querySelector("#record-dialog");
let editing = null;

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

function showMessage(text) {
  message.textContent = text;
  window.setTimeout(() => { if (message.textContent === text) message.textContent = ""; }, 4000);
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}

function recordRow(title, detail, badge = "", buttons = "") {
  return `<article class="record"><div class="record-main"><div class="record-title">${title}</div><div class="muted">${detail}</div></div>${badge ? `<span class="badge ${esc(badge)}">${esc(badge.replaceAll("_", " "))}</span>` : ""}<div class="actions">${buttons}</div></article>`;
}

function actionButton(action, id, label) {
  return `<button data-action="${action}" data-id="${id}">${label}</button>`;
}

function setView(name) {
  document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab.dataset.view === name));
  document.querySelectorAll(".view").forEach(view => view.classList.toggle("hidden", view.id !== `${name}-view`));
  if (name === "loans") loadLoans();
}

const schemas = {
  student: [ ["name", "Full name", "text"], ["email", "Email", "email"], ["phone", "Phone", "tel"] ],
  staff: [ ["name", "Full name", "text"], ["email", "Email", "email"], ["role", "Role", "text"] ],
  book: [ ["title", "Title", "text"], ["author", "Author", "text"], ["isbn", "ISBN", "text"], ["year_published", "Year published", "number"] ]
};

function openForm(type, record = null) {
  editing = record ? { type, id: record.id } : { type, id: null };
  document.querySelector("#dialog-title").textContent = `${record ? "Edit" : "Add"} ${type}`;
  document.querySelector("#dialog-fields").innerHTML = schemas[type].map(([key, label, inputType]) =>
    `<label>${label}<input name="${key}" type="${inputType}" value="${esc(record?.[key] ?? "")}" required></label>`
  ).join("");
  dialog.showModal();
}

async function loadBooks() {
  const search = encodeURIComponent(document.querySelector("#search").value);
  const books = await api(`/api/books?search=${search}`);
  document.querySelector("#books").innerHTML = books.map(book => recordRow(
    `${esc(book.title)} <span class="muted">(${book.year_published})</span>`,
    `${esc(book.author)} · ISBN ${esc(book.isbn)}`,
    book.status,
    `${book.status === "available" ? actionButton("borrow", book.id, "Borrow") : ""}${actionButton("edit-book", book.id, "Edit")}${actionButton("delete-book", book.id, "Delete")}`
  )).join("") || "<p class='muted'>No books match this search.</p>";
}

async function loadMembers() {
  const [students, staff, books] = await Promise.all([
    api("/api/students"), api("/api/staff"), api("/api/books")
  ]);
  document.querySelector("#loan-student").innerHTML = students.map(person => `<option value="${person.id}">${esc(person.name)}</option>`).join("");
  document.querySelector("#loan-staff").innerHTML = staff.map(person => `<option value="${person.id}">${esc(person.name)}</option>`).join("");
  document.querySelector("#loan-book").innerHTML = books.filter(book => book.status === "available")
    .map(book => `<option value="${book.id}">${esc(book.title)}</option>`).join("");
}

async function loadStudents() {
  const search = encodeURIComponent(document.querySelector("#student-search").value);
  const students = await api(`/api/students?search=${search}`);
  document.querySelector("#students").innerHTML = students.map(student => recordRow(
    esc(student.name), `${esc(student.email)} · ${esc(student.phone || "No phone")}`,
    "", `${actionButton("edit-student", student.id, "Edit")}${actionButton("delete-student", student.id, "Delete")}`
  )).join("") || "<p class='muted'>No students found.</p>";
}

async function loadStaff() {
  const staff = await api("/api/staff");
  document.querySelector("#staff").innerHTML = staff.map(person => recordRow(
    esc(person.name), `${esc(person.email)} · ${esc(person.role)}`,
    "", `${actionButton("edit-staff", person.id, "Edit")}${actionButton("delete-staff", person.id, "Delete")}`
  )).join("") || "<p class='muted'>No staff records.</p>";
}

async function loadLoans() {
  const [loans, overdue] = await Promise.all([api("/api/loans"), api("/api/loans/overdue")]);
  document.querySelector("#loans").innerHTML = loans.map(loan => recordRow(
    esc(loan.book_title), `Student: ${esc(loan.student_name)} · Issued by: ${esc(loan.staff_name)} · Due: ${esc(loan.due_on)}`,
    loan.status, loan.status !== "returned" ? actionButton("return-loan", loan.id, "Return") : ""
  )).join("") || "<p class='muted'>No loans yet.</p>";
  document.querySelector("#overdue").innerHTML = overdue.map(loan => recordRow(
    esc(loan.title), `Student: ${esc(loan.student_name)} · Due: ${esc(loan.due_on)}`, "overdue"
  )).join("") || "<p class='muted'>No overdue loans.</p>";
}

async function refreshAll() {
  try {
    await Promise.all([loadBooks(), loadStudents(), loadStaff(), loadMembers()]);
    if (!document.querySelector("#loans-view").classList.contains("hidden")) await loadLoans();
  } catch (error) { showMessage(error.message); }
}

document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
document.querySelector("#search").addEventListener("input", () => loadBooks().catch(error => showMessage(error.message)));
document.querySelector("#student-search").addEventListener("input", () => loadStudents().catch(error => showMessage(error.message)));
document.querySelector("#add-book").addEventListener("click", () => openForm("book"));
document.querySelector("#add-student").addEventListener("click", () => openForm("student"));
document.querySelector("#add-staff").addEventListener("click", () => openForm("staff"));
document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
document.querySelector("#cancel-dialog").addEventListener("click", () => dialog.close());

document.querySelector("#record-form").addEventListener("submit", async event => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.currentTarget));
  if (editing.type === "book") data.year_published = Number(data.year_published);
  const plural = { student: "students", staff: "staff", book: "books" }[editing.type];
  const path = `/api/${plural}${editing.id ? `/${editing.id}` : ""}`;
  try {
    await api(path, { method: editing.id ? "PUT" : "POST", body: JSON.stringify(data) });
    dialog.close(); await refreshAll(); showMessage(`${editing.type} saved.`);
  } catch (error) { showMessage(error.message); }
});

document.querySelector("#loan-form").addEventListener("submit", async event => {
  event.preventDefault();
  const payload = {
    book_id: Number(document.querySelector("#loan-book").value),
    student_id: Number(document.querySelector("#loan-student").value),
    staff_id: Number(document.querySelector("#loan-staff").value)
  };
  try {
    const loan = await api("/api/loans", { method: "POST", body: JSON.stringify(payload) });
    showMessage(`Book checked out. Due ${loan.due_on}.`); await refreshAll(); await loadLoans();
  } catch (error) { showMessage(error.message); }
});

document.querySelector("main").addEventListener("click", async event => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const id = Number(button.dataset.id), action = button.dataset.action;
  try {
    if (action.startsWith("edit-")) {
      const type = action.slice(5);
      const endpoint = type === "book" ? "books" : type === "student" ? "students" : "staff";
      const records = await api(`/api/${endpoint}`);
      const record = records.find(item => item.id === id);
      if (record) openForm(type, record);
      return;
    }
    if (action.startsWith("delete-")) {
      const type = action.slice(7), endpoint = type === "book" ? "books" : type === "student" ? "students" : "staff";
      if (!confirm(`Delete this ${type}? Records used in loan history cannot be deleted.`)) return;
      await api(`/api/${endpoint}/${id}`, { method: "DELETE" });
      showMessage(`${type} deleted.`); await refreshAll(); return;
    }
    if (action === "return-loan") {
      await api(`/api/loans/${id}/return`, { method: "PATCH" });
      showMessage("Book returned."); await refreshAll(); await loadLoans();
    }
    if (action === "borrow") {
      await loadMembers();
      setView("loans");
      document.querySelector("#loan-book").value = String(id);
      showMessage("Choose a student and staff member, then check out the book.");
    }
  } catch (error) { showMessage(error.message); }
});

refreshAll();
```

#### 6. Run and explore

Open three terminals from the project root or the relevant folders:

1. **PostgreSQL:** from the project root, run `docker compose up -d`.
2. **FastAPI:** from `backend`, activate the virtual environment and run `uvicorn app.main:app --reload`. Visit `http://localhost:8000/docs` to inspect and try the API.
3. **Frontend:** from `frontend`, run `python -m http.server 5500`, then visit `http://localhost:5500`.

This setup assumes a fresh demo database. If PostgreSQL already has tables from an earlier version of this lesson, stop the compose service and reset its named volume with `docker compose down -v`, then start it again with `docker compose up -d`. This removes the old demo data so the updated table definitions can be created cleanly.

Students can create a student, book, or staff record from the matching dashboard tab, edit it, and try to delete it. A record referenced by loan history is protected by the foreign key and cannot be deleted. Then search the catalog, choose an available title, student, and issuing staff member, and check out the book. They should observe the `POST` request in the browser's developer tools Network panel, inspect its JSON response, and see the badge update after the next `GET`. Return the book with the **Return** button in the Loans tab; the frontend sends `PATCH`, then refreshes the history and catalog. To explore overdue results, change a loan's `due_on` to yesterday in the database and refresh the loan view. This activity makes each CRUD action visible while keeping the focus on how data moves across the layers.

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*
