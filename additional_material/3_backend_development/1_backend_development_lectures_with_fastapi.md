### Backend Development Lectures with FastAPI


---





## 1. The Foundations of Web Communication & Local Setup

### 1) Client–Server Model (What actually happens?)
- **Client**: A browser/mobile app/another server makes a request.
- **Server**: Your FastAPI app receives the request, runs code, and returns a response.
- **Stateless**: Each request is independent; servers don’t remember previous requests unless you store state (DB, cache, JWTs, etc.).

For example, a browser requests a product, FastAPI looks it up in a database, and the server returns the result. The database is optional: some requests can be answered without one. A stateless request carries the information needed to process it, such as an authentication token; the application can still store persistent data in a database.

Visual: Client → Internet → Web Server (Uvicorn/Gunicorn) → App (FastAPI) → DB (PostgreSQL) → App → Client.

### 2) Communication Protocols Overview

Protocols are rules for communication. Different layers handle different jobs: transport moves data, while application protocols define how applications exchange and interpret messages.

- **TCP** is a connection-oriented transport protocol that delivers an ordered byte stream and retransmits lost data. It is used by HTTP/1.1 and HTTP/2, as well as many database connections.
- **UDP** is a transport protocol that sends independent datagrams without guaranteeing delivery or order. It is useful for latency-sensitive applications such as live audio and gaming; DNS also commonly uses it. Applications can add reliability when needed.
- **HTTP** is an application protocol: it defines requests and responses, including methods, headers, and status codes. **HTTPS** protects HTTP with TLS, which encrypts traffic, protects its integrity, and authenticates the server.
- **WebSockets** provide a persistent, bidirectional connection for cases such as chat and live dashboards.
- **gRPC** is an RPC framework commonly used for typed service-to-service communication and streaming, typically over HTTP/2 with Protocol Buffers.

Think of HTTP as the language of a conversation, TLS as a sealed envelope, and TCP or QUIC as the delivery service. HTTP/1.1 and HTTP/2 normally use TCP. HTTP/3 uses QUIC over UDP; QUIC adds reliability and TLS-based security. Use HTTPS for production APIs regardless of HTTP version.

### 3) Deep Dive into HTTP/HTTPS

An HTTP request contains a method and path, optional headers, and sometimes a body. The server answers with a status code, headers, and optionally a body. HTTPS describes the security of this exchange; it is separate from the HTTP version:

| Version | How it communicates |
| --- | --- |
| HTTP/1.1 | Usually uses TCP; browsers may open several connections for concurrent requests. |
| HTTP/2 | Usually uses TCP and multiplexes multiple requests over one connection; it also compresses headers. |
| HTTP/3 | Uses QUIC over UDP, allowing independent streams to avoid TCP's connection-wide blocking when packets are lost. |

These versions retain the same core HTTP concepts. A browser and server negotiate a mutually supported version; a newer version is not guaranteed for every connection. HTTP/3 includes encryption, while HTTPS can also be used with HTTP/1.1 or HTTP/2.

In a React + FastAPI app, React's `fetch` or Axios call does not choose the HTTP version. The browser and the server or proxy it contacts determine it. Locally, a browser connecting directly to Uvicorn normally uses HTTP/1.1. In production, a configured reverse proxy may accept HTTP/2 or HTTP/3 from the browser and forward HTTP/1.1 to Uvicorn. Inspect the **Protocol** column in the browser's Network tools to see the version used for a request.

- **Methods**: GET (read), POST (create), PUT/PATCH (update), DELETE (remove)
- **Status Codes**: 200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 404 Not Found, 422 Unprocessable Entity, 500 Internal Server Error
- **Headers**: `Content-Type`, `Authorization`, `Accept`, `Cache-Control`
- **Bodies**: Typically JSON for APIs

Quick demo using curl:
```bash
curl -i https://httpbin.org/get
curl -i -X POST https://httpbin.org/post \
  -H "Content-Type: application/json" \
  -d '{"message":"hello"}'
```

### HTTP in practice: follow one request

For `GET https://shop.example.com/products/12?currency=USD`, DNS resolves the host, the client establishes a connection and TLS for HTTPS, then sends a request with a method, path, query and headers. The server responds with a status, headers and possibly a body. TLS protects data in transit; it does not decide which user may access a resource.

| Situation | Typical response | Why |
| --- | --- | --- |
| Created a resource | 201, optionally a `Location` header | A new resource exists |
| Invalid input | 422 in FastAPI | Request shape or values failed validation |
| Missing/invalid credentials | 401 | Authenticate first |
| Authenticated but forbidden | 403 | Identity lacks permission |
| Missing resource | 404 | No matching accessible resource |
| Unique constraint conflict | 409 | Request conflicts with existing state |
| Unexpected server failure | 500 | Log details privately; return a safe message |

GET should not change server state. PUT replaces the represented fields; PATCH changes selected fields. GET, PUT and DELETE are intended to be idempotent: repeating the same request has the same intended effect, although responses can differ. POST is not generally idempotent; payment and order APIs often use an idempotency key to make retries safe. Network timeouts are ambiguous: the server might have completed the operation even if the client never received the response.

**Checkpoint:** identify the method, path, query, request body and response status for creating a product; explain why a timeout should not always trigger a blind POST retry.

### 4) APIs & REST (Principles)
- **Resource-oriented**: `/products`, `/orders`, `/employees`
- **Stateless**: Server doesn’t store client session state between requests
- **Uniform interface**: Predictable URLs, methods, and status codes
- **Representation**: JSON by default; use schemas for validation

### 5) Local Development Environment
Install Python 3.10+ and Git.

```bash
# On Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" sqlalchemy "psycopg[binary]" pydantic pydantic-settings alembic python-dotenv email-validator
```

Recommended files to start with:
```bash
@('fastapi', 'uvicorn[standard]', 'sqlalchemy', 'psycopg[binary]', 'pydantic', 'pydantic-settings', 'alembic', 'python-dotenv', 'email-validator') | Set-Content requirements.txt
@('__pycache__/', '.venv/', '.env') | Set-Content .gitignore
```

### 6) Version Control
```bash
git init
git add .
git commit -m "chore: initial setup"
```

You can connect a remote repository later if you choose to deploy.

---

## 2. Building Your First Web Server

### 1) Web Server vs Web Framework
- **Web server** (Uvicorn/Gunicorn): Listens on a port, speaks HTTP protocol.
- **Framework** (FastAPI): Routing, validation, dependency injection, docs.

### 2) Routing: Map URLs to Code
Create `main.py` with a minimal FastAPI app and one route.

```python
# main.py
from fastapi import FastAPI

app = FastAPI(title="Hello FastAPI")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is running!"}
```

Run locally:
```bash
uvicorn main:app --reload --port 8000
# Visit http://localhost:8000 and http://localhost:8000/docs
```

### 3) Handling Requests: Path, Query, Headers, Body

```python
# app_requests.py (examples; you can merge into main.py if preferred)
from typing import Optional
from fastapi import FastAPI, Header
from pydantic import BaseModel

app = FastAPI(title="Request Handling Examples")

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

@app.get("/items/{item_id}")
def get_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "query": q}

@app.get("/headers")
def read_headers(user_agent: Optional[str] = Header(None)):
    return {"user_agent": user_agent}

@app.post("/items")
def create_item(item: Item):
    return {"status": "created", "item": item}
```

Key ideas:
- Type hints drive validation and docs.
- Path params: in the route template.
- Query params: regular function params with defaults.
- Headers: use `Header`.
- Body: Pydantic models.

### 4) Sending Responses (JSON by default)
Returning a `dict` automatically becomes JSON.

```python
@app.get("/health")
def health():
    return {"ok": True}
```



---

### Practice: validation and errors

Try `/items/abc` and observe FastAPI's 422 response; try `/items/1?q=pen` and compare path and query values. Add a `GET /items/{item_id}` branch that raises `HTTPException(status_code=404, detail="Item not found")` for unknown IDs. Add a Pydantic `Field(gt=0)` for price and try zero, a negative value, a missing field and malformed JSON. Inspect the generated schema at `/docs`. Response models are an output boundary: define public fields explicitly so internal fields do not leak.

**Checkpoint:** explain why a missing record and an invalid integer produce different responses. Avoid returning stack traces or secrets to clients.

## 3. Database Integration with PostgreSQL

### 1) RDBMS Fundamentals (PostgreSQL)
- **Schemas, Tables, Columns, Rows**
- **Primary Keys, Foreign Keys**
- **Relationships**: one-to-many, many-to-many

### 2) Load DATABASE_URL securely with Pydantic Settings
Install and start PostgreSQL locally, create a development database (for example, `createdb inventory_db`), and create `.env` locally; do not commit it. Use your local PostgreSQL username and password. If `createdb` is unavailable, create the database using pgAdmin or `psql`.

```
# .env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/inventory_db
```

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", env_file_encoding="utf-8")

settings = Settings()
```

### 3) SQLAlchemy Engine, Session, and Models

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

```python
# models.py
from sqlalchemy import Column, Integer, String, Numeric, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from database import Base

class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True)
```

```python
# schemas.py
from pydantic import BaseModel, condecimal

class ProductCreate(BaseModel):
    name: str
    price: condecimal(max_digits=10, decimal_places=2)
    in_stock: bool = True

class ProductRead(BaseModel):
    id: int
    name: str
    price: condecimal(max_digits=10, decimal_places=2)
    in_stock: bool

    class Config:
        from_attributes = True
```

### 4) CRUD Endpoints (Live DB)

```python
# main.py (CRUD edition)
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import Base, engine, get_db
from models import Product
from schemas import ProductCreate, ProductRead

app = FastAPI(title="Products API")
Base.metadata.create_all(bind=engine)

@app.post("/products", response_model=ProductRead, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    exists = db.query(Product).filter(Product.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=400, detail="Product with this name already exists")
    product = Product(name=payload.name, price=payload.price, in_stock=payload.in_stock)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product

@app.get("/products/{product_id}", response_model=ProductRead)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.get("/products", response_model=list[ProductRead])
def list_products(db: Session = Depends(get_db)):
    return db.query(Product).order_by(Product.id).all()

@app.put("/products/{product_id}", response_model=ProductRead)
def update_product(product_id: int, payload: ProductCreate, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    product.name = payload.name
    product.price = payload.price
    product.in_stock = payload.in_stock
    db.commit()
    db.refresh(product)
    return product

@app.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    db.delete(product)
    db.commit()
    return None
```

### 5)  Dependencies - try to do this on your own

```python
# requirements.txt
fastapi
uvicorn
SQLAlchemy
pydantic-settings
psycopg
psycopg[binary]
alembic
python-dotenv
```

Test locally:
```bash
uvicorn main:app --reload
curl -X POST http://localhost:8000/products -H "Content-Type: application/json" -d '{"name":"Pen","price":"1.50","in_stock":true}'
curl http://localhost:8000/products
```



### Database reasoning and edge cases

A primary key identifies a row; a foreign key enforces a reference; a unique constraint protects an invariant even under concurrent requests. An index can speed reads but costs space and write time. Use `Decimal` with `Numeric` for money, and validate nonnegative prices and stock. A transaction groups changes: commit all successful changes or roll back on failure. After a failed flush or commit, call `rollback()` before reusing the session. A preliminary duplicate check is helpful feedback but is not concurrency-safe: retain the database unique constraint and catch the relevant `IntegrityError`, roll back, and return a safe 409. Do not label every integrity error as a duplicate without checking its cause.

A list endpoint should eventually support a bounded `limit`, `offset` or cursor, stable ordering, filtering and documented defaults. For PATCH use a separate model with optional fields and `model_dump(exclude_unset=True)` so omitted fields differ from explicit nulls; validate nullability before applying updates. Use parameterized SQL or ORM expressions, never string interpolation of user input.

**Practice:** test create, list, get, replace and delete; then test nonexistent IDs, duplicate names, invalid prices and an empty list. Explain what happens if two clients create the same name at once. Add pagination and a partial update as extensions.

### 5) Database Migrations with Alembic
Initialize Alembic and configure it to use your SQLAlchemy `Base` metadata.

```bash
alembic init alembic
```

Edit `alembic.ini` to use `sqlalchemy.url` via env variable (optional), and `alembic/env.py` to target your metadata:

```python
# alembic/env.py (excerpt)
from sqlalchemy import create_engine, pool
from alembic import context
import os
from logging.config import fileConfig
from database import Base


from dotenv import load_dotenv  # <-- add this
load_dotenv()  # <-- this line loads your .env file

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_online():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set")

    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()

```

Import `models` in `alembic/env.py` before using `Base.metadata` so Alembic sees the tables. The excerpt above shows online migrations only; retain Alembic's generated offline/online dispatch if both modes are required. For an existing database, `create_all` does not update columns and can make a first autogenerated migration empty. Choose migrations as the schema source of truth: remove `create_all` for the migration workflow and start with a fresh development database, or carefully establish a reviewed baseline for an existing one. Review generated revisions before applying them.

Create and apply migrations:
```bash
alembic revision --autogenerate -m "create products table"
alembic upgrade head
```

## Project 1: Mini E‑Commerce Inventory API (FastAPI + PostgreSQL)

Goal: Build a small, production-like API with proper DB models and CRUD. The app tracks products and employees and allows simple inventory operations.

### Project acceptance criteria and stretch cases

Build each resource first, then connect it to the database and test it. Demonstrate valid CRUD, missing IDs, duplicate email/name, invalid stock and an empty database. Add database constraints for nonnegative quantity and price; document whether deletion is hard or soft. A stock adjustment should be a transaction, and concurrent decrements need an atomic conditional update or locking to avoid overselling. Document setup, example requests, migrations, and how to run tests. Stretch: add pagination, an audit trail, and role-based access before deployment.

### Features
- Manage products: create/list/get/update/delete
- Manage employees: create/list/get/update/delete
- Track stock levels on products (simple integer quantity)

### Data Model
- `Product`: `id`, `name`, `price`, `quantity` (integer stock), `is_active`
- `Employee`: `id`, `full_name`, `email`, `is_active`

### Suggested Project Structure
```
.
├─ main.py
├─ config.py
├─ database.py
├─ models.py
├─ schemas.py
├─ routers/
│  ├─ products.py
│  └─ employees.py
├─ alembic/
├─ requirements.txt
└─ Procfile
```

### Code: Models and Schemas
```python
# models.py
from sqlalchemy import Integer, String, Boolean, Numeric
from sqlalchemy.orm import Mapped, mapped_column
from database import Base

class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

```python
# schemas.py
from pydantic import BaseModel, EmailStr, condecimal

class ProductBase(BaseModel):
    name: str
    price: condecimal(max_digits=10, decimal_places=2)
    quantity: int = 0
    is_active: bool = True

class ProductCreate(ProductBase):
    pass

class ProductRead(ProductBase):
    id: int
    class Config:
        from_attributes = True

class EmployeeBase(BaseModel):
    full_name: str
    email: EmailStr
    is_active: bool = True

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeRead(EmployeeBase):
    id: int
    class Config:
        from_attributes = True
```

### Code: Routers
```python
# routers/products.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Product
from schemas import ProductCreate, ProductRead

router = APIRouter(prefix="/products", tags=["products"])

@router.post("", response_model=ProductRead, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    if db.query(Product).filter(Product.name == payload.name).first():
        raise HTTPException(400, "Product name already exists")
    product = Product(**payload.dict())
    db.add(product)
    db.commit()
    db.refresh(product)
    return product

@router.get("/{product_id}", response_model=ProductRead)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    return product

@router.get("", response_model=list[ProductRead])
def list_products(db: Session = Depends(get_db)):
    return db.query(Product).order_by(Product.id).all()

@router.put("/{product_id}", response_model=ProductRead)
def update_product(product_id: int, payload: ProductCreate, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    for field, value in payload.dict().items():
        setattr(product, field, value)
    db.commit()
    db.refresh(product)
    return product

@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    db.delete(product)
    db.commit()
    return None
```

```python
# routers/employees.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Employee
from schemas import EmployeeCreate, EmployeeRead

router = APIRouter(prefix="/employees", tags=["employees"])

@router.post("", response_model=EmployeeRead, status_code=status.HTTP_201_CREATED)
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db)):
    if db.query(Employee).filter(Employee.email == payload.email).first():
        raise HTTPException(400, "Email already exists")
    employee = Employee(**payload.dict())
    db.add(employee)
    db.commit()
    db.refresh(employee)
    return employee

@router.get("/{employee_id}", response_model=EmployeeRead)
def get_employee(employee_id: int, db: Session = Depends(get_db)):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    return emp

@router.get("", response_model=list[EmployeeRead])
def list_employees(db: Session = Depends(get_db)):
    return db.query(Employee).order_by(Employee.id).all()

@router.put("/{employee_id}", response_model=EmployeeRead)
def update_employee(employee_id: int, payload: EmployeeCreate, db: Session = Depends(get_db)):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    for field, value in payload.dict().items():
        setattr(emp, field, value)
    db.commit()
    db.refresh(emp)
    return emp

@router.delete("/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_employee(employee_id: int, db: Session = Depends(get_db)):
    emp = db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    db.delete(emp)
    db.commit()
    return None
```

### Code: App Entrypoint
```python
# main.py
from fastapi import FastAPI
from database import Base, engine
from routers import products, employees

app = FastAPI(title="Mini E‑Commerce Inventory API")
Base.metadata.create_all(bind=engine)

app.include_router(products.router)
app.include_router(employees.router)

@app.get("/health")
def health():
    return {"ok": True}
```

### Local migrations and startup
Initialize Alembic and generate migrations after defining models. If you already initialized Alembic above, reuse that configuration rather than initializing it again.
```bash
alembic init alembic
alembic revision --autogenerate -m "init products and employees"
alembic upgrade head
```

Start the API locally from the project directory:
```bash
uvicorn main:app --reload
```
Open `http://127.0.0.1:8000/docs` to try the endpoints. Keep the server running while testing.

### Quick local tests

Using the interactive docs, create a product with `{"name":"Notebook","price":"3.99","quantity":25}` and an employee with `{"full_name":"Jane Doe","email":"jane@example.com"}`. List each resource, then try reading, updating, and deleting a record. Check the missing-record and invalid-input cases too. The API runs at `http://127.0.0.1:8000`.

At this point the local application is complete. The following deployment steps are optional.

## Optional: Cloud deployment

### Optional: Set up Railway
Prereqs: A GitHub account.

- Create an account on Railway (`https://railway.app`).
- Create a new project.
- Add a **PostgreSQL** service.
- Note the connection string (will be provided as `DATABASE_URL`).
- Install Railway CLI:
```bash
npm i -g @railway/cli
railway login
```


Push your project to a GitHub repository if you want Railway to deploy it from GitHub. Keep `.env` and credentials out of the repository. Railway's interface and configuration can change, so check its current documentation when deploying.

### Optional: Deploy the API and database on Railway
- Set `DATABASE_URL` in Railway service settings (copy from the Railway Postgres plugin).
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Run Alembic migrations in a Railway shell or via a deploy hook.

Optional `Procfile`:
```bash
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

### Optional: Vercel for a frontend

This project builds an API and does not include a frontend, so Vercel is not needed to run it. If you later build a separate frontend, you can deploy that frontend to Vercel and configure it to call the deployed API. Allow the exact frontend origin in the API's CORS settings, use HTTPS, and configure the API URL as a frontend environment variable. Follow Vercel's current deployment instructions for your frontend framework.
