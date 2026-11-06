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
Install Python 3.10+, Git, and Docker Desktop (or Docker Engine with the Compose plugin). Start Docker before the PostgreSQL setup in section 3.

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

### 2) Start PostgreSQL locally with Docker Compose

Create `compose.yaml` in your working directory. The API and Alembic run on your computer; only PostgreSQL runs in the container.

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: inventory_user
      POSTGRES_PASSWORD: local_dev_password
      POSTGRES_DB: inventory_db
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U inventory_user -d inventory_db"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  postgres_data:
```

These credentials are for local practice only. From the directory containing `compose.yaml`, run:

```bash
docker compose up -d
docker compose ps
docker compose exec db pg_isready -U inventory_user -d inventory_db
```

Wait until PostgreSQL is ready before starting the API. The image creates the user and database on first initialization; no separate `createdb` is needed. The named volume retains data across container restarts. `docker compose down` stops and removes the container while retaining the volume. Changing the initial user, password, or database in the file does not alter an already initialized volume. If port 5432 is occupied, change the mapping to `127.0.0.1:5433:5432` and use port `5433` in the host-side database URL below. Use `docker compose logs db` to diagnose startup problems.

### 3) Load DATABASE_URL securely with Pydantic Settings

Create `.env` next to `compose.yaml` and `config.py`. It is already included in `.gitignore`; do not commit it. These values match the Compose configuration above:

```
# .env
DATABASE_URL=postgresql+psycopg://inventory_user:local_dev_password@localhost:5432/inventory_db
```

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", env_file_encoding="utf-8")

settings = Settings()
```

### 4) SQLAlchemy Engine, Session, and Models

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

### 5) CRUD Endpoints (Live DB)

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

### 6) Dependencies - try to do this on your own

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
email-validator
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

### 7) Database Migrations with Alembic
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

For this migration exercise, stop Uvicorn, remove `Base.metadata.create_all(bind=engine)` from the CRUD `main.py`, and create a fresh database in the same container:

```bash
docker compose exec db createdb -U inventory_user inventory_migrations_db
```

Change the database name in the local `.env` URL from `inventory_db` to `inventory_migrations_db` (retain the same credentials and host port). The original database and its data remain untouched. Create and apply migrations:
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

### Prepare a separate project database

The earlier CRUD example has a different `products` table from this project. Keep the running Compose database service and create a second, empty database inside it:

```bash
docker compose exec db createdb -U inventory_user inventory_project_db
```

For the inventory project, use a new project directory with its own `.env`, `models.py`, `schemas.py`, `main.py`, and Alembic directory. Copy `config.py` and `database.py` from the earlier example and set the new project's `.env` to:

```text
DATABASE_URL=postgresql+psycopg://inventory_user:local_dev_password@localhost:5432/inventory_project_db
```

If you changed the host port to 5433, use that port here too. Continue running Compose commands from the directory containing `compose.yaml`; run the API and Alembic commands from the new project directory. The databases share the same local container and volume but have separate schemas and migration histories.

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
├─ .env                 # local only; do not commit
└─ .gitignore
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
from routers import products, employees

app = FastAPI(title="Mini E‑Commerce Inventory API")

app.include_router(products.router)
app.include_router(employees.router)

@app.get("/health")
def health():
    return {"ok": True}
```

### Local migrations and startup
Initialize Alembic in the new project directory after defining both models. Configure `alembic/env.py` as shown above, importing the new project’s `models` and using its `.env`. Keep this migration history separate from the earlier example. Review the generated revision before applying it. The project entrypoint does not call `create_all`; migrations create the tables.
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

### Build the frontend: Inventory Studio

Create a `frontend` folder in the inventory project and add the three files below. This dashboard uses plain HTML, CSS, and JavaScript to manage the real products and employees stored by your API. It includes search, live summary cards, responsive tables, and a shared create/edit dialog. No build tools or external libraries are required.

```text
frontend/
├─ index.html
├─ styles.css
└─ app.js
```

#### Serve the frontend from FastAPI

Add these imports to `main.py`:

```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles
```

Append this mount **after** the existing API routes and health endpoint:

```python
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
```

Keep the existing router registrations. The mount comes last so `/products`, `/employees`, `/health`, and `/docs` keep working. Requests from the page use the same origin as the API, so this setup needs no CORS configuration. Create all three frontend files before starting the server; do not open `index.html` directly with `file://`.

#### 1. HTML — `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#152c26">
  <title>Inventory Studio</title>
  <link rel="stylesheet" href="/styles.css">
  <script src="/app.js" defer></script>
</head>
<body>
  <aside class="sidebar">
    <a class="brand" href="/" aria-label="Inventory Studio home">
      <span class="brand-icon">i.</span><span>Inventory<span class="brand-light">Studio</span></span>
    </a>
    <p class="nav-label">WORKSPACE</p>
    <nav aria-label="Resource navigation">
      <button class="nav-item selected" data-view="products" aria-pressed="true">▦ <span>Products</span></button>
      <button class="nav-item" data-view="employees" aria-pressed="false">◎ <span>Employees</span></button>
    </nav>
    <div class="sidebar-note"><span class="small-dot"></span> Your local workspace<p>Small details.<br>Better operations.</p></div>
    <a class="docs-link" href="/docs" target="_blank" rel="noopener">Explore API docs ↗</a>
  </aside>

  <main>
    <header class="topbar"><span>Workspace <span class="slash">/</span> Overview</span><span class="avatar" aria-label="Inventory Studio">IS</span></header>
    <section class="hero">
      <div><p class="eyebrow">A LITTLE ORDER GOES A LONG WAY</p><h1>Your inventory,<br><em>in good hands.</em></h1><p class="hero-copy">A clear view of your products and the people behind them.</p></div>
      <div class="hero-art" aria-hidden="true"><div class="orbit"></div><div class="parcel">▦</div><span class="art-label">EVERYTHING IN ITS PLACE</span></div>
    </section>

    <section class="stats" aria-label="Workspace totals">
      <article class="stat"><span>Total products</span><strong id="product-count">—</strong><small>Your complete catalog</small></article>
      <article class="stat"><span>Inventory value</span><strong id="inventory-value">—</strong><small>Price × units in stock · USD</small></article>
      <article class="stat"><span>Low stock</span><strong id="low-stock">—</strong><small>Products with 5 units or fewer</small></article>
      <article class="stat"><span>Team members</span><strong id="employee-count">—</strong><small>People in your workspace</small></article>
    </section>

    <section class="panel" aria-labelledby="table-title">
      <div class="panel-heading"><div><p class="eyebrow">THE DETAILS</p><h2 id="table-title">Product catalog</h2></div><button id="add-button" class="button primary" disabled>+ Add product</button></div>
      <div class="toolbar"><label class="search"><span aria-hidden="true">⌕</span><input id="search" type="search" placeholder="Search products…" aria-label="Search records"></label><button id="refresh" class="button subtle">↻ Refresh</button></div>
      <p id="notice" class="notice" role="status" aria-live="polite">Connecting to your API…</p>
      <div class="table-scroll" aria-busy="true" id="table-region"><table><caption class="sr-only" id="table-caption">Products</caption><thead id="table-head"></thead><tbody id="table-body"></tbody></table></div>
      <div class="panel-footer"><span id="result-count">Loading records…</span><span>Made for a more organized day.</span></div>
    </section>
    <footer class="page-footer">INVENTORY STUDIO <span>Built with FastAPI + a little frontend magic.</span></footer>
  </main>

  <dialog id="editor" aria-labelledby="editor-title">
    <form id="record-form">
      <div class="dialog-heading"><div><p class="eyebrow">YOUR WORKSPACE</p><h2 id="editor-title">Add product</h2></div><button type="button" id="close-dialog" class="icon-button" aria-label="Close dialog">×</button></div>
      <div id="fields"></div>
      <p id="form-error" class="form-error" role="alert"></p>
      <div class="dialog-actions"><button type="button" id="cancel-dialog" class="button subtle">Cancel</button><button type="submit" id="save-button" class="button primary">Save record</button></div>
    </form>
  </dialog>
</body>
</html>
```

#### 2. CSS — `frontend/styles.css`

```css
:root { color-scheme: light; --ink: #213c33; --muted: #718078; --paper: #f7f8f2; --line: #e4e8de; --green: #254f3e; --lime: #d9edaa; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, sans-serif; }
button, input { font: inherit; }
button, a { -webkit-tap-highlight-color: transparent; }
button { cursor: pointer; }
a { color: inherit; }
button:disabled { opacity: .5; cursor: wait; }
:focus-visible { outline: 3px solid #729c41; outline-offset: 4px; }
.sidebar { position: fixed; inset: 0 auto 0 0; width: 236px; padding: 36px 24px; background: #152c26; color: #f4f7ef; display: flex; flex-direction: column; }
.brand { display: flex; align-items: center; gap: 12px; text-decoration: none; font-size: 19px; font-weight: 700; letter-spacing: -.6px; }
.brand-icon { display: grid; place-items: center; background: var(--lime); color: #152c26; width: 40px; height: 40px; border-radius: 13px; font: italic 32px Georgia, serif; }
.brand-light { display: block; color: #a8bcb0; font-size: 13px; font-weight: 400; letter-spacing: 2px; }
.nav-label { font-size: 10px; letter-spacing: 2px; color: #93aa9c; margin: 56px 12px 15px; }
.nav-item { width: 100%; border: 0; background: transparent; color: #b8c9bf; text-align: left; padding: 14px 18px; border-radius: 10px; margin-bottom: 8px; font-size: 20px; display: flex; align-items: center; gap: 14px; }
.nav-item span { font-size: 14px; }
.nav-item.selected { background: var(--lime); color: #213c33; }
.sidebar-note { margin: auto 10px 26px; font-size: 11px; color: #b2c4b8; }
.small-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--lime); margin-right: 6px; }
.sidebar-note p { font: 26px/1.3 Georgia, serif; color: #e0e9dc; }
.docs-link { border-top: 1px solid #355046; padding: 20px 10px 0; font-size: 12px; text-decoration: none; }
main { margin-left: 236px; padding: 0 48px; max-width: 1700px; }
.topbar { height: 88px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--muted); }
.slash { padding: 0 12px; color: #b8c2b6; }
.avatar { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 50%; background: #e5eadb; color: var(--green); font-weight: 700; }
.hero { display: flex; justify-content: space-between; align-items: center; padding: 42px 0 36px; gap: 20px; }
.eyebrow { font-size: 10px; font-weight: 700; letter-spacing: 2px; color: #7b8c68; margin: 0 0 12px; }
h1 { font-size: clamp(34px, 3.5vw, 54px); line-height: 1.1; font-weight: 500; letter-spacing: -2px; margin: 0; }
h1 em { font-family: Georgia, serif; color: #75904e; font-weight: 400; }
.hero-copy { color: var(--muted); font-size: 13px; margin: 20px 0 0; }
.hero-art { position: relative; width: 240px; height: 180px; display: grid; place-items: center; flex-shrink: 0; }
.orbit { position: absolute; width: 170px; height: 170px; border: 1px solid #d8dfca; border-radius: 50%; background: radial-gradient(circle, #e3edcc 0, transparent 70%); }
.parcel { z-index: 1; font-size: 100px; line-height: 1; color: #577647; transform: rotate(-13deg); text-shadow: 12px 14px 0 #d6e3bd; }
.art-label { position: absolute; bottom: 0; font-size: 8px; letter-spacing: 2px; color: #7b8c68; }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-bottom: 32px; }
.stat { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 22px; }
.stat span, .stat small { display: block; color: var(--muted); font-size: 12px; }
.stat strong { display: block; font-size: clamp(22px, 2.4vw, 34px); font-weight: 500; letter-spacing: -1px; margin: 12px 0 8px; overflow-wrap: anywhere; }
.stat small { font-size: 10px; }
.stat:nth-child(3) { background: #edf2df; }
.panel { border: 1px solid var(--line); border-radius: 16px; background: #fff; overflow: hidden; }
.panel-heading { padding: 26px 26px 20px; display: flex; justify-content: space-between; align-items: center; gap: 16px; }
h2 { font-size: 22px; font-weight: 500; letter-spacing: -.7px; margin: 0; }
.panel-heading .eyebrow { margin-bottom: 6px; }
.button { border: 1px solid transparent; border-radius: 8px; padding: 10px 16px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.primary { background: var(--green); color: #fff; }
.primary:hover { background: #37694e; }
.subtle { background: #fff; border-color: var(--line); color: var(--ink); }
.subtle:hover { background: #f1f4e9; }
.toolbar { padding: 0 26px 18px; display: flex; gap: 12px; justify-content: space-between; }
.search { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border: 1px solid var(--line); border-radius: 8px; width: min(360px, 100%); color: var(--muted); }
.search input { border: 0; outline: 0; width: 100%; min-width: 0; background: transparent; font-size: 12px; color: var(--ink); }
.search:focus-within { outline: 2px solid #729c41; }
.notice { margin: 0; padding: 0 26px 16px; font-size: 12px; color: var(--muted); }
.notice.error, .form-error { color: #a63434; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; text-align: left; white-space: nowrap; }
th { background: #f8faf5; color: #7a877a; text-transform: uppercase; letter-spacing: 1px; font-size: 9px; font-weight: 600; }
th, td { padding: 16px 26px; border-bottom: 1px solid #edf0e8; }
td { font-size: 12px; }
tbody tr:hover { background: #fcfdf9; }
.record-name { font-weight: 600; max-width: 280px; white-space: normal; overflow-wrap: anywhere; }
.badge { display: inline-block; border-radius: 20px; background: #eaf3e6; color: #49723d; padding: 4px 10px; font-size: 10px; }
.badge.inactive { background: #f0f0ec; color: #7a8178; }
.badge.low { background: #fff2da; color: #936a23; }
.row-action { border: 0; background: transparent; color: var(--green); padding: 6px 8px; font-size: 11px; border-radius: 5px; }
.row-action:hover { background: #eef3e7; }
.row-action.danger { color: #a94848; }
.empty { text-align: center; color: var(--muted); padding: 48px 20px; white-space: normal; }
.panel-footer { padding: 16px 26px; display: flex; justify-content: space-between; gap: 12px; font-size: 10px; color: var(--muted); }
.page-footer { display: flex; justify-content: space-between; gap: 16px; padding: 26px 0; font-size: 9px; letter-spacing: 1px; color: #899481; }
.page-footer span { letter-spacing: 0; }
dialog { border: 1px solid var(--line); border-radius: 18px; width: min(480px, calc(100% - 32px)); padding: 28px; color: var(--ink); box-shadow: 0 30px 100px #152c2633; max-height: 90vh; overflow-y: auto; }
dialog::backdrop { background: #152c2677; backdrop-filter: blur(4px); }
.dialog-heading { display: flex; justify-content: space-between; align-items: start; margin-bottom: 24px; }
.icon-button { border: 0; background: #f0f3e9; border-radius: 50%; width: 30px; height: 30px; font-size: 21px; color: var(--ink); }
.field { display: grid; gap: 7px; font-size: 12px; margin-bottom: 16px; }
.field input { width: 100%; padding: 11px 12px; border: 1px solid #dce2d4; border-radius: 8px; color: var(--ink); background: #fcfdf9; }
.checkbox { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.checkbox input { accent-color: var(--green); }
.form-error { font-size: 12px; white-space: pre-line; }
.dialog-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 24px; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
@media (min-width: 1600px) { main { margin-right: auto; } }
@media (max-width: 1150px) { .sidebar { width: 200px; padding: 28px 18px; } main { margin-left: 200px; padding: 0 28px; } .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 720px) { .sidebar { position: static; width: 100%; padding: 18px 20px; flex-direction: row; align-items: center; gap: 18px; flex-wrap: wrap; } .brand { font-size: 16px; } .brand-light { font-size: 10px; } nav { display: flex; gap: 6px; margin-left: auto; } .nav-item { width: auto; margin: 0; padding: 8px 10px; gap: 6px; } .nav-item span { font-size: 11px; } .nav-label, .sidebar-note, .docs-link { display: none; } main { margin: 0; padding: 0 20px; } .topbar { height: 60px; } .hero { padding: 30px 0; } .hero-art { display: none; } h1 { font-size: 40px; } .stats { gap: 10px; margin-bottom: 20px; } .stat { padding: 16px; } .panel-heading { padding: 20px 16px; } .toolbar { padding: 0 16px 16px; } th, td { padding: 14px 16px; } .notice { padding-left: 16px; padding-right: 16px; } .panel-footer { padding: 14px 16px; } .panel-footer span:last-child, .page-footer span { display: none; } }
```

#### 3. JavaScript — `frontend/app.js`

The API returns decimal prices as strings or numbers; `Number()` handles both for display. All record text is inserted with `textContent`, so names and emails are treated as text. The form sends every required field on `PUT`, matching the routers above.

```javascript
const $ = (selector) => document.querySelector(selector);
const state = { view: "products", products: [], employees: [], ready: false, busy: false, editing: null };
const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const labels = {
  products: ["Product", "Price", "Stock", "Status", "Actions"],
  employees: ["Team member", "Email", "Status", "Actions"],
};

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Cannot reach the API. Check that Uvicorn is running, then refresh.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item) => `${item.loc.join(".")}: ${item.msg}`).join("\n")
      : body.detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status}). Check the API terminal for details.`);
  }
  return response.status === 204 ? null : response.json();
}

function notice(message, error = false) {
  $("#notice").textContent = message;
  $("#notice").classList.toggle("error", error);
}

function setBusy(busy) {
  state.busy = busy;
  $("#refresh").disabled = busy;
  $("#add-button").disabled = busy || !state.ready;
  $("#save-button").disabled = busy;
  $("#close-dialog").disabled = busy;
  $("#cancel-dialog").disabled = busy;
  document.querySelectorAll(".row-action").forEach((button) => { button.disabled = busy; });
  $("#table-region").setAttribute("aria-busy", String(busy));
}

async function loadData(message = "Workspace is up to date.") {
  // Commit both lists together so totals and tables describe one completed refresh.
  const [products, employees] = await Promise.all([api("/products"), api("/employees")]);
  state.products = products;
  state.employees = employees;
  state.ready = true;
  render();
  notice(message);
}

async function refresh() {
  setBusy(true);
  notice("Refreshing your workspace…");
  try { await loadData(); }
  catch (error) { notice(error.message + (state.ready ? " Showing previously loaded records." : ""), true); }
  finally { setBusy(false); }
}

function cell(row, value, className = "") {
  const td = document.createElement("td");
  td.textContent = value;
  td.className = className;
  row.append(td);
  return td;
}

function badge(td, text, kind = "") {
  const span = document.createElement("span");
  span.className = `badge ${kind}`;
  span.textContent = text;
  td.append(span);
}

function render() {
  const isProduct = state.view === "products";
  $("#product-count").textContent = state.products.length;
  $("#employee-count").textContent = state.employees.length;
  $("#inventory-value").textContent = money.format(state.products.reduce((total, p) => total + Number(p.price) * p.quantity, 0));
  $("#low-stock").textContent = state.products.filter((p) => p.quantity <= 5).length;
  $("#table-title").textContent = isProduct ? "Product catalog" : "Your team";
  $("#table-caption").textContent = isProduct ? "Products" : "Employees";
  $("#add-button").textContent = isProduct ? "+ Add product" : "+ Add employee";
  $("#search").placeholder = isProduct ? "Search products…" : "Search name or email…";
  document.querySelectorAll("[data-view]").forEach((button) => {
    const selected = button.dataset.view === state.view;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  const heading = document.createElement("tr");
  labels[state.view].forEach((label) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    heading.append(th);
  });
  $("#table-head").replaceChildren(heading);
  const query = $("#search").value.trim().toLowerCase();
  const records = state[state.view].filter((record) =>
    (isProduct ? record.name : `${record.full_name} ${record.email}`).toLowerCase().includes(query)
  );
  $("#table-body").replaceChildren();
  for (const record of records) {
    const row = document.createElement("tr");
    cell(row, isProduct ? record.name : record.full_name, "record-name");
    if (isProduct) {
      cell(row, money.format(Number(record.price)));
      badge(cell(row, ""), `${record.quantity} units`, record.quantity <= 5 ? "low" : "");
    } else { cell(row, record.email); }
    badge(cell(row, ""), record.is_active ? "Active" : "Inactive", record.is_active ? "" : "inactive");
    const actions = cell(row, "");
    for (const action of ["Edit", "Delete"]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = action;
      button.className = `row-action ${action === "Delete" ? "danger" : ""}`;
      button.disabled = state.busy;
      button.setAttribute("aria-label", `${action} ${isProduct ? record.name : record.full_name}`);
      button.addEventListener("click", () => action === "Edit" ? openEditor(record) : deleteRecord(record));
      actions.append(button);
    }
    $("#table-body").append(row);
  }
  if (!records.length) {
    const row = document.createElement("tr");
    const td = cell(row, query ? "No matching records. Try a different search." : "A fresh start. Add your first record to bring this workspace to life.", "empty");
    td.colSpan = labels[state.view].length;
    $("#table-body").append(row);
  }
  $("#result-count").textContent = `${records.length} of ${state[state.view].length} ${state.view}`;
}

function openEditor(record = null) {
  if (state.busy || !state.ready) return;
  // Capture the resource now; navigation can change the visible table later.
  state.editing = { resource: state.view, id: record?.id ?? null };
  const isProduct = state.view === "products";
  $("#editor-title").textContent = `${record ? "Edit" : "Add"} ${isProduct ? "product" : "employee"}`;
  $("#fields").replaceChildren();
  $("#form-error").textContent = "";
  const definitions = isProduct
    ? [["name", "Product name", "text"], ["price", "Unit price (USD)", "number"], ["quantity", "Quantity in stock", "number"]]
    : [["full_name", "Full name", "text"], ["email", "Email address", "email"]];
  for (const [name, title, type] of definitions) {
    const label = document.createElement("label");
    label.className = "field";
    label.textContent = title;
    const input = document.createElement("input");
    input.name = name;
    input.type = type;
    input.required = true;
    input.value = record?.[name] ?? (type === "number" ? "0" : "");
    if (type === "number") {
      input.min = "0";
      input.step = name === "price" ? "0.01" : "1";
      input.max = name === "price" ? "99999999.99" : "2147483647";
    } else { input.maxLength = name === "email" ? 320 : 200; }
    label.append(input);
    $("#fields").append(label);
  }
  const activeLabel = document.createElement("label");
  activeLabel.className = "checkbox";
  const active = document.createElement("input");
  active.type = "checkbox";
  active.name = "is_active";
  active.checked = record?.is_active ?? true;
  activeLabel.append(active, document.createTextNode("Active record"));
  $("#fields").append(activeLabel);
  $("#editor").showModal();
}

$("#record-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const { resource, id } = state.editing;
  const data = new FormData(event.currentTarget);
  const payload = { is_active: data.has("is_active") };
  if (resource === "products") {
    payload.name = data.get("name").trim();
    payload.price = data.get("price");
    payload.quantity = Number(data.get("quantity"));
  } else {
    payload.full_name = data.get("full_name").trim();
    payload.email = data.get("email").trim();
  }
  if ((resource === "products" && !payload.name) || (resource === "employees" && !payload.full_name)) {
    $("#form-error").textContent = "Enter a name containing at least one non-space character.";
    return;
  }
  setBusy(true);
  $("#form-error").textContent = "";
  try {
    await api(`/${resource}${id === null ? "" : `/${id}`}`, {
      method: id === null ? "POST" : "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("#editor").close();
    try { await loadData("Record saved successfully."); }
    catch (error) { notice(`Record saved, but the refresh failed. ${error.message} Refresh before making another change.`, true); }
  } catch (error) { $("#form-error").textContent = error.message; }
  finally { setBusy(false); }
});

async function deleteRecord(record) {
  if (state.busy) return;
  const resource = state.view;
  const name = resource === "products" ? record.name : record.full_name;
  if (!window.confirm(`Permanently delete "${name}"? This cannot be undone.`)) return;
  setBusy(true);
  try {
    await api(`/${resource}/${record.id}`, { method: "DELETE" });
    try { await loadData("Record deleted."); }
    catch (error) { notice(`Record deleted, but the refresh failed. ${error.message} Refresh before making another change.`, true); }
  } catch (error) { notice(error.message, true); }
  finally { setBusy(false); }
}

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    $("#search").value = "";
    if (state.ready) render();
  });
});
$("#search").addEventListener("input", () => { if (state.ready) render(); });
$("#refresh").addEventListener("click", refresh);
$("#add-button").addEventListener("click", () => openEditor());
for (const selector of ["#close-dialog", "#cancel-dialog"]) {
  $(selector).addEventListener("click", () => { if (!state.busy) $("#editor").close(); });
}
$("#editor").addEventListener("cancel", (event) => { if (state.busy) event.preventDefault(); });
refresh();
```

#### Run and try the dashboard

Restart `uvicorn main:app --reload` if needed, then open **http://127.0.0.1:8000/**. Keep the API and PostgreSQL container running.

1. Add a product such as Notebook, price `3.99`, quantity `25`. The catalog and summary cards should update after saving.
2. Edit its quantity to `3` to see the low-stock badge and count change. Search for it by name.
3. Switch to Employees, add Jane Doe with `jane@example.com`, then edit her active status.
4. Delete a record and confirm that it disappears. These endpoints permanently delete records.
5. Try a duplicate product name or employee email when creating a record to see the API's error inside the dialog. Stop the API and press Refresh to check the connection error, then restart it and refresh again.
6. Resize the browser to a phone width and navigate the dialog using the keyboard. Wide tables scroll horizontally on small screens.

The totals include active and inactive records. Inventory value uses USD as a display convention; change the currency in `Intl.NumberFormat` and the visible labels if your project uses another currency. This is a local learning dashboard: the current API has no authentication. Browser form constraints improve usability, but the API must also enforce nonnegative quantities/prices and handle duplicate-name/email updates as described in the acceptance criteria above.

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
- Set `DATABASE_URL` in Railway service settings to the Railway PostgreSQL connection string, using the `postgresql+psycopg://` SQLAlchemy scheme. The local Compose hostname and credentials do not work in Railway.
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Run Alembic migrations in a Railway shell or via a deploy hook.

Optional `Procfile`:
```bash
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

### Optional: Vercel for a frontend

The frontend above is served directly by FastAPI, so Vercel is not needed to run it. If you later deploy the frontend separately to Vercel, replace the relative API paths in `app.js` with your deployed API base URL and configure the frontend to call that API. Allow the exact frontend origin in the API's CORS settings, use HTTPS, and configure the API URL as a frontend environment variable. Follow Vercel's current deployment instructions for your frontend framework.
