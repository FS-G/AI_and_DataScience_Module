### Backend Development Lectures with FastAPI (Modules 1–3 + Capstone 1)

This curriculum is hands-on and cloud-first. Each module includes short theory, practical steps, and copy-pasteable code you can run locally and deploy to the cloud (Railway + PostgreSQL).

---

## Module 1: The Foundations of Web Communication & Cloud Setup

### 1) Client–Server Model (What actually happens?)
- **Client**: A browser/mobile app/another server makes a request.
- **Server**: Your FastAPI app receives the request, runs code, and returns a response.
- **Stateless**: Each request is independent; servers don’t remember previous requests unless you store state (DB, cache, JWTs, etc.).

Visual: Client → Internet → Web Server (Uvicorn/Gunicorn) → App (FastAPI) → DB (PostgreSQL) → App → Client.

### 2) Communication Protocols Overview
- **TCP**: Reliable, ordered, connection-oriented. Used under HTTP/HTTPS.
  - When to use: Most web apps and APIs; anything that must not lose messages (e.g., REST APIs, database connections).
- **UDP**: Fast, unreliable, used for real-time streaming/gaming.
  - When to use: Low-latency, loss-tolerant scenarios (live audio/video, gaming telemetry, DNS). Not for typical APIs.
- **HTTP/HTTPS**: Application protocol of the web; resources, methods, headers.
  - When to use: Standard request/response APIs, browser interactions, public endpoints; use HTTPS in production, always.
- **WebSockets**: Full-duplex, persistent connection for realtime apps.
  - When to use: Real-time updates and bidirectional messaging (chat, live dashboards, notifications) where server can push instantly.
- **gRPC**: High-performance RPC over HTTP/2 with protobuf schemas.
  - When to use: Service-to-service communication, strongly typed contracts, streaming RPCs, and high throughput microservices.

### 3) Deep Dive into HTTP/HTTPS
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

### 4) APIs & REST (Principles)
- **Resource-oriented**: `/products`, `/orders`, `/employees`
- **Stateless**: Server doesn’t store client session state between requests
- **Uniform interface**: Predictable URLs, methods, and status codes
- **Representation**: JSON by default; use schemas for validation

### 5) Setting Up Your Cloud-Native Environment (Railway)
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

### 6) Local Development Environment
Install Python 3.10+ and Git.

```bash
# On Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install fastapi uvicorn[standard] sqlalchemy psycopg[binary] pydantic pydantic-settings alembic python-dotenv
```

Recommended files to start with:
```bash
echo "fastapi\nuvicorn[standard]\nsqlalchemy\npsycopg[binary]\npydantic\npydantic-settings\nalembic\npython-dotenv" > requirements.txt
echo "__pycache__/\n.venv/\n.env" > .gitignore
```

### 7) Version Control & Initial Deployment Prep
```bash
git init
git add .
git commit -m "chore: initial setup"
gh repo create yourname/fastapi-starter --public --source=. --remote=origin --push
```

Link to Railway and set env later after we have an app. Railway supports push-to-deploy from GitHub.

---

## Module 2: Building Your First Web Server

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

## Module 3: Cloud Database Integration with PostgreSQL

### 1) RDBMS Fundamentals (PostgreSQL)
- **Schemas, Tables, Columns, Rows**
- **Primary Keys, Foreign Keys**
- **Relationships**: one-to-many, many-to-many

### 2) Load DATABASE_URL securely with Pydantic Settings
Create `.env` locally (do not commit) and set on Railway as an environment variable.

```
# .env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME
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

Create and apply migrations:
```bash
alembic revision --autogenerate -m "create products table"
alembic upgrade head
```

### 6) Deploy with Railway (DB + App)
- Set `DATABASE_URL` in Railway service settings (copy from the Railway Postgres plugin).
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Run Alembic migrations in a Railway shell or via a deploy hook.

Optional `Procfile`:
```bash
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Capstone Project 1: Mini E‑Commerce Inventory API (FastAPI + PostgreSQL)

Goal: Build a small, production-like API with proper DB models and CRUD. The app tracks products and employees and allows simple inventory operations.

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

### Migrations and Deployment
1) Initialize Alembic and generate migrations after defining models
```bash
alembic init alembic
alembic revision --autogenerate -m "init products and employees"
alembic upgrade head
```

2) Railway Deployment
- Add your repo as a Railway service.
- Set env var `DATABASE_URL` from the Railway PostgreSQL plugin.
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT` (via service settings or `Procfile`).
- Run Alembic migrations in a Railway shell.

### Quick Test Commands
```bash
# Products
curl -X POST $URL/products -H "Content-Type: application/json" -d '{"name":"Notebook","price":"3.99","quantity":25}'
curl $URL/products

# Employees
curl -X POST $URL/employees -H "Content-Type: application/json" -d '{"full_name":"Jane Doe","email":"jane@example.com"}'
curl $URL/employees
```

That’s it—by the end of Module 3 and this capstone, you will have a real, cloud-deployed FastAPI + PostgreSQL application with migrations, clean routing, and production-ready structure.

