**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*

---

# PostgreSQL Demo App: Step-by-Step Build Guide

## Project Structure

```text
postgres-demo-app/
├── compose.yaml
├── backend/
│   ├── requirements.txt
│   ├── main.py
│   ├── .env
│   └── Dockerfile
└── frontend/
    ├── index.html
    ├── styles.css
    └── app.js
```

## DB — PostgreSQL

### Step 1: Create `compose.yaml`

```yaml
services:
  db:
    image: postgres:17
    container_name: shop-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: shop_db
      POSTGRES_USER: shop_user
      POSTGRES_PASSWORD: shop_pass_123
    ports:
      - "5432:5432"
    volumes:
      - shop_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U shop_user -d shop_db"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  shop_data:
```

### Step 2: Start the Database

```bash
docker compose up -d db
docker compose ps
```

### Step 3: Connect in DBeaver

```text
Host: localhost
Port: 5432
Database: shop_db
Username: shop_user
Password: shop_pass_123
```

### Step 4: Verify in DBeaver

```sql
SELECT current_database(), current_user;
```

### Step 1: Create `backend/requirements.txt`

```text
fastapi==0.115.12
uvicorn[standard]==0.34.2
psycopg[binary]==3.2.9
python-dotenv==1.2.4
```

### Step 2: Create `backend/main.py`

```python
import os
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))
DATABASE_URL = os.environ["DATABASE_URL"]


def connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name VARCHAR(120) NOT NULL,
                description TEXT,
                price NUMERIC(10, 2) NOT NULL CHECK (price >= 0),
                stock INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0),
                active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
                status VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'paid', 'shipped', 'cancelled')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                order_item_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
                product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE RESTRICT,
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
                UNIQUE (order_id, product_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id)")
    yield


app = FastAPI(title="Shop Demo API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=255)


class ProductIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None
    price: Decimal = Field(ge=0, max_digits=10, decimal_places=2)
    stock: int = Field(default=0, ge=0)


class OrderItemIn(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)


class OrderIn(BaseModel):
    customer_id: int
    items: list[OrderItemIn] = Field(min_length=1)


class OrderStatusIn(BaseModel):
    status: str = Field(pattern="^(pending|paid|shipped|cancelled)$")


@app.get("/api/health")
def health():
    with connect() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/api/customers")
def list_customers():
    with connect() as conn:
        return conn.execute("SELECT * FROM customers ORDER BY customer_id").fetchall()


@app.post("/api/customers", status_code=201)
def create_customer(body: CustomerIn):
    try:
        with connect() as conn:
            return conn.execute(
                "INSERT INTO customers (name, email) VALUES (%s, %s) RETURNING *",
                (body.name, body.email),
            ).fetchone()
    except psycopg.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="Email already exists")


@app.put("/api/customers/{customer_id}")
def update_customer(customer_id: int, body: CustomerIn):
    try:
        with connect() as conn:
            row = conn.execute(
                "UPDATE customers SET name = %s, email = %s WHERE customer_id = %s RETURNING *",
                (body.name, body.email, customer_id),
            ).fetchone()
    except psycopg.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="Email already exists")
    if row is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return row


@app.delete("/api/customers/{customer_id}", status_code=204)
def delete_customer(customer_id: int):
    try:
        with connect() as conn:
            result = conn.execute(
                "DELETE FROM customers WHERE customer_id = %s", (customer_id,)
            )
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status_code=409, detail="Customer has orders and cannot be deleted")
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    return Response(status_code=204)


@app.get("/api/products")
def list_products():
    with connect() as conn:
        return conn.execute("SELECT * FROM products ORDER BY product_id").fetchall()


@app.post("/api/products", status_code=201)
def create_product(body: ProductIn):
    with connect() as conn:
        return conn.execute(
            """INSERT INTO products (name, description, price, stock)
               VALUES (%s, %s, %s, %s) RETURNING *""",
            (body.name, body.description, body.price, body.stock),
        ).fetchone()


@app.put("/api/products/{product_id}")
def update_product(product_id: int, body: ProductIn):
    with connect() as conn:
        row = conn.execute(
            """UPDATE products SET name = %s, description = %s, price = %s, stock = %s
               WHERE product_id = %s RETURNING *""",
            (body.name, body.description, body.price, body.stock, product_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return row


@app.delete("/api/products/{product_id}", status_code=204)
def delete_product(product_id: int):
    try:
        with connect() as conn:
            result = conn.execute("DELETE FROM products WHERE product_id = %s", (product_id,))
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status_code=409, detail="Product is used by an order")
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return Response(status_code=204)


@app.get("/api/orders")
def list_orders():
    with connect() as conn:
        return conn.execute("""
            SELECT o.order_id, o.customer_id, c.name AS customer_name, o.status,
                   o.created_at, COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total
            FROM orders o
            JOIN customers c ON c.customer_id = o.customer_id
            LEFT JOIN order_items oi ON oi.order_id = o.order_id
            GROUP BY o.order_id, c.name
            ORDER BY o.order_id DESC
        """).fetchall()


@app.post("/api/orders", status_code=201)
def create_order(body: OrderIn):
    try:
        with connect() as conn:
            with conn.transaction():
                order = conn.execute(
                    "INSERT INTO orders (customer_id) VALUES (%s) RETURNING *",
                    (body.customer_id,),
                ).fetchone()
                for item in body.items:
                    product = conn.execute(
                        "SELECT product_id, price, stock FROM products WHERE product_id = %s FOR UPDATE",
                        (item.product_id,),
                    ).fetchone()
                    if product is None:
                        raise HTTPException(status_code=404, detail=f"Product {item.product_id} not found")
                    if product["stock"] < item.quantity:
                        raise HTTPException(status_code=409, detail=f"Insufficient stock for product {item.product_id}")
                    conn.execute(
                        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (%s, %s, %s, %s)",
                        (order["order_id"], item.product_id, item.quantity, product["price"]),
                    )
                    conn.execute(
                        "UPDATE products SET stock = stock - %s WHERE product_id = %s",
                        (item.quantity, item.product_id),
                    )
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status_code=404, detail="Customer not found")
    except psycopg.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="A product may appear only once per order")
    return {"order_id": order["order_id"], "customer_id": order["customer_id"], "status": order["status"]}


@app.patch("/api/orders/{order_id}/status")
def update_order_status(order_id: int, body: OrderStatusIn):
    with connect() as conn:
        row = conn.execute(
            "UPDATE orders SET status = %s WHERE order_id = %s RETURNING *",
            (body.status, order_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return row


@app.delete("/api/orders/{order_id}", status_code=204)
def delete_order(order_id: int):
    with connect() as conn:
        result = conn.execute("DELETE FROM orders WHERE order_id = %s", (order_id,))
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    return Response(status_code=204)
```

### Option A: Run FastAPI Directly

Choose this option to run the backend directly in your terminal. First start PostgreSQL:

```bash
docker compose up -d db
```

When the backend runs directly on your computer, the database host is `localhost`. Create **`backend/.env`** in the same folder as `main.py` with this exact line:

```env
DATABASE_URL=postgresql://shop_user:shop_pass_123@localhost:5432/shop_db
```

From the `backend` folder, create and activate a virtual environment and install the dependencies.

Windows PowerShell:

```powershell
py -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Because `main.py` loads `backend/.env`, run this from the `backend` folder on any platform:

```bash
uvicorn main:app --reload
```

Keep the PostgreSQL container running while using the locally launched API.


### Option B: Run FastAPI with Docker Compose

The two backend options are alternatives: **choose Option A or Option B**. Do not run both at once because each uses port `8000`. For this option, Compose supplies `DATABASE_URL` from the API service configuration below. Inside Compose, the database hostname is `db`, which Docker resolves on the Compose network.

#### Step 1: Create `backend/Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 2: Add the API Service to `compose.yaml`

```yaml
  api:
    build: ./backend
    container_name: shop-api
    restart: unless-stopped
    environment:
      DATABASE_URL: postgresql://shop_user:shop_pass_123@db:5432/shop_db
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
```

Add this `api` service under `services:` in `compose.yaml`, at the same indentation level as `db`.

#### Step 3: Build and Start the API with Docker Compose

```bash
docker compose up -d --build
docker compose ps
```

#### Step 4: Open the API Docs

```text
http://localhost:8000/docs
```

#### Step 5: Verify Backend and Tables

```text
http://localhost:8000/api/health
```

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

## FRONTEND — HTML, CSS, JavaScript

### Step 1: Create `frontend/index.html`

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shop Demo</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="#top"><span class="brand-mark">S</span> Shopdesk</a>
    <span class="environment"><span class="live-dot"></span> Demo workspace</span>
  </header>
  <main class="container" id="top">
    <section class="welcome">
      <div><p class="eyebrow">STORE MANAGEMENT</p><h1>Good day, team</h1><p class="muted">Manage your customers, catalog, and orders in one place.</p></div>
      <button class="button button-primary" type="button" onclick="document.querySelector('#customer-name').focus()">＋ Add customer</button>
    </section>

    <div class="stats-grid" aria-label="Store summary">
      <article class="stat-card"><span class="stat-icon lavender">♙</span><div><p>Total customers</p><strong id="customer-count">—</strong></div></article>
      <article class="stat-card"><span class="stat-icon mint">▦</span><div><p>Active products</p><strong id="product-count">—</strong></div></article>
      <article class="stat-card"><span class="stat-icon peach">▤</span><div><p>Total orders</p><strong id="order-count">—</strong></div></article>
      <article class="stat-card"><span class="stat-icon blue">$</span><div><p>Order value</p><strong id="order-value">—</strong></div></article>
    </div>

    <section class="panel" id="customer-section">
      <div class="panel-heading"><div><p class="eyebrow">PEOPLE</p><h2>Customers</h2><p class="muted">Create and maintain customer records.</p></div></div>
      <form id="customer-form" class="form-grid">
        <label>Name<input id="customer-name" name="name" placeholder="e.g. Aisha Khan" required maxlength="100"></label>
        <label>Email<input name="email" type="email" placeholder="name@example.com" required maxlength="255"></label>
        <div class="form-actions"><button id="customer-submit" class="button button-primary">Add customer</button><button id="customer-cancel" class="button button-light" type="button" hidden>Cancel</button></div>
      </form>
      <div class="table-wrap"><table><thead><tr><th>Customer</th><th>Email</th><th>Joined</th><th class="align-right">Actions</th></tr></thead><tbody id="customers-table"></tbody></table></div>
      <p id="customers-empty" class="empty-state" hidden>No customers yet. Add your first customer above.</p>
    </section>

    <section class="panel">
      <div class="panel-heading"><div><p class="eyebrow">CATALOG</p><h2>Products</h2><p class="muted">Track pricing and available stock.</p></div></div>
      <form id="product-form" class="form-grid product-form">
        <label>Product name<input id="product-name" name="name" placeholder="e.g. Wireless Mouse" required maxlength="120"></label>
        <label>Description<input name="description" placeholder="Short product description"></label>
        <label>Price<input name="price" type="number" min="0" step="0.01" placeholder="0.00" required></label>
        <label>Stock quantity<input name="stock" type="number" min="0" step="1" value="0" required></label>
        <div class="form-actions"><button id="product-submit" class="button button-primary">Add product</button><button id="product-cancel" class="button button-light" type="button" hidden>Cancel</button></div>
      </form>
      <div class="table-wrap"><table><thead><tr><th>Product</th><th>Price</th><th>Stock</th><th class="align-right">Actions</th></tr></thead><tbody id="products-table"></tbody></table></div>
      <p id="products-empty" class="empty-state" hidden>No products yet. Add your first product above.</p>
    </section>

    <section class="panel">
      <div class="panel-heading"><div><p class="eyebrow">SALES</p><h2>Orders</h2><p class="muted">Create an order, update its status, or remove it.</p></div></div>
      <form id="order-form" class="order-form">
        <label>Customer<select name="customer_id" id="order-customer" required><option value="">Choose a customer</option></select></label>
        <label>Product<select name="product_id" id="order-product" required><option value="">Choose a product</option></select></label>
        <label class="quantity-field">Quantity<input name="quantity" type="number" min="1" step="1" value="1" required></label>
        <button class="button button-primary">Create order</button>
      </form>
      <div class="table-wrap"><table><thead><tr><th>Order</th><th>Customer</th><th>Date</th><th>Total</th><th>Status</th><th class="align-right">Actions</th></tr></thead><tbody id="orders-table"></tbody></table></div>
      <p id="orders-empty" class="empty-state" hidden>No orders yet. Create an order above.</p>
    </section>
    <div id="toast" class="toast" role="status" aria-live="polite"></div>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

### Step 2: Create `frontend/styles.css`

```css
* { box-sizing: border-box; }
:root { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f5f7fb; font-synthesis: none; }
body { margin: 0; min-width: 320px; font-size: 14px; line-height: 1.5; }
button, input, select { font: inherit; }
.topbar { height: 66px; padding: 0 max(24px, calc((100vw - 1120px) / 2)); display: flex; align-items: center; justify-content: space-between; background: #fff; border-bottom: 1px solid #e9edf4; }
.brand { display: flex; align-items: center; gap: 10px; color: #172033; text-decoration: none; font-size: 17px; font-weight: 750; letter-spacing: -.4px; }
.brand-mark { display: grid; place-items: center; width: 30px; height: 30px; color: #fff; background: #5368e8; border-radius: 9px; }
.environment { color: #667085; font-size: 12px; }.live-dot { display: inline-block; width: 7px; height: 7px; margin-right: 7px; background: #25a879; border-radius: 50%; }
.container { width: min(1120px, calc(100% - 40px)); margin: 34px auto 70px; }
.welcome { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 25px; }.welcome h1 { margin: 3px 0 4px; font-size: clamp(25px, 3vw, 32px); letter-spacing: -1px; }.muted { margin: 3px 0 0; color: #778198; }
.eyebrow { margin: 0 0 4px; color: #74809a; font-size: 10px; font-weight: 750; letter-spacing: 1.25px; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 22px; }.stat-card { display: flex; align-items: center; gap: 13px; min-height: 90px; padding: 17px; background: #fff; border: 1px solid #edf0f5; border-radius: 12px; }.stat-icon { display: grid; place-items: center; flex: 0 0 42px; height: 42px; border-radius: 12px; font-size: 20px; font-weight: 700; }.lavender { color: #5d58ca; background: #eeedff; }.mint { color: #16866a; background: #e3f8f1; }.peach { color: #c26b37; background: #fff0e5; }.blue { color: #3a73c5; background: #e9f2ff; }.stat-card p { margin: 0; color: #778198; font-size: 12px; }.stat-card strong { display: block; margin-top: 2px; font-size: 22px; letter-spacing: -.5px; }
.panel { margin: 17px 0; padding: 22px 24px; background: #fff; border: 1px solid #e9edf4; border-radius: 13px; box-shadow: 0 3px 12px #1d2d4b05; }.panel-heading { display: flex; justify-content: space-between; margin-bottom: 18px; }.panel h2 { margin: 0; font-size: 19px; letter-spacing: -.35px; }
.form-grid { display: grid; grid-template-columns: 1fr 1fr auto; align-items: end; gap: 12px; margin-bottom: 20px; padding: 15px; background: #f8f9fc; border: 1px solid #edf0f5; border-radius: 10px; }.product-form { grid-template-columns: 1.2fr 1.5fr .7fr .8fr auto; }
label { display: grid; gap: 6px; color: #535f75; font-size: 12px; font-weight: 650; }input, select { width: 100%; min-width: 0; height: 39px; padding: 0 11px; color: #263247; background: #fff; border: 1px solid #dce2ec; border-radius: 7px; outline: none; }input:focus, select:focus { border-color: #7584f0; box-shadow: 0 0 0 3px #5368e81c; }input::placeholder { color: #a1a9b8; }
.form-actions { display: flex; gap: 7px; }.button { min-height: 38px; padding: 0 13px; border: 1px solid transparent; border-radius: 7px; cursor: pointer; font-size: 12px; font-weight: 700; white-space: nowrap; }.button-primary { color: #fff; background: #5368e8; }.button-primary:hover { background: #4358d4; }.button-light { color: #59657a; background: #fff; border-color: #dce2ec; }.welcome .button { min-height: 40px; padding: 0 16px; }
.table-wrap { width: 100%; overflow-x: auto; }table { width: 100%; border-collapse: collapse; text-align: left; white-space: nowrap; }th { padding: 10px 12px; color: #8992a4; background: #fafbfc; font-size: 10px; font-weight: 750; letter-spacing: .65px; text-transform: uppercase; }td { padding: 12px; border-bottom: 1px solid #eef1f5; color: #47536a; }tbody tr:last-child td { border-bottom: 0; }.primary-cell { color: #222d42; font-weight: 650; }.secondary-cell { display: block; color: #8a93a3; font-size: 11px; }.align-right { text-align: right; }
.actions { display: flex; justify-content: flex-end; gap: 6px; }.text-button { padding: 5px 8px; color: #5264d8; background: transparent; border: 0; border-radius: 5px; cursor: pointer; font-size: 12px; font-weight: 650; }.text-button:hover { background: #f0f2ff; }.text-button.delete { color: #bd4c59; }.text-button.delete:hover { background: #fff0f1; }
.badge { display: inline-flex; align-items: center; padding: 4px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: capitalize; }.badge.pending { color: #a66a13; background: #fff4db; }.badge.paid { color: #27734f; background: #e3f6eb; }.badge.shipped { color: #3864aa; background: #e9f1ff; }.badge.cancelled { color: #9b4d56; background: #ffeaec; }.stock-low { color: #ba5b40; font-weight: 700; }
.order-form { display: grid; grid-template-columns: 1fr 1fr 120px auto; align-items: end; gap: 12px; margin-bottom: 20px; padding: 15px; background: #f8f9fc; border: 1px solid #edf0f5; border-radius: 10px; }.empty-state { padding: 18px; color: #8a93a3; text-align: center; }.toast { position: fixed; right: 24px; bottom: 24px; z-index: 5; max-width: min(400px, calc(100vw - 48px)); padding: 12px 16px; color: #fff; background: #25314a; border-radius: 9px; box-shadow: 0 8px 30px #15213a30; opacity: 0; pointer-events: none; transform: translateY(8px); transition: .2s ease; }.toast.show { opacity: 1; transform: translateY(0); }.toast.error { background: #a93645; }
[hidden] { display: none !important; }
@media (max-width: 900px) { .stats-grid { grid-template-columns: repeat(2, 1fr); }.product-form { grid-template-columns: repeat(2, 1fr); }.product-form .form-actions { grid-column: 1 / -1; } }
@media (max-width: 620px) { .topbar { padding: 0 18px; }.container { width: calc(100% - 28px); margin-top: 24px; }.welcome { align-items: flex-start; flex-direction: column; }.stats-grid { gap: 9px; }.stat-card { min-height: 78px; padding: 12px; gap: 9px; }.stat-icon { flex-basis: 34px; height: 34px; }.panel { padding: 17px 14px; }.form-grid, .product-form, .order-form { grid-template-columns: 1fr; }.form-actions { grid-column: auto !important; }.environment { font-size: 11px; } }
```

### Step 3: Create `frontend/app.js`

```javascript
const API = "http://localhost:8000/api";
const toast = document.querySelector("#toast");
let customers = [];
let products = [];
let editingCustomerId = null;
let editingProductId = null;
let toastTimer;

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

function showMessage(text, isError = false) {
  toast.textContent = text;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3500);
}

function cell(text, className = "") {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function actionButton(label, className, handler) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `text-button ${className}`.trim();
  button.textContent = label;
  button.addEventListener("click", handler);
  return button;
}

function emptyRow(tbody, colspan, text) {
  const tr = document.createElement("tr");
  const td = cell(text);
  td.colSpan = colspan;
  td.className = "empty-state";
  tr.append(td);
  tbody.append(tr);
}

function setEditMode(kind, record = null) {
  const isCustomer = kind === "customer";
  const form = document.querySelector(isCustomer ? "#customer-form" : "#product-form");
  const id = isCustomer ? "customer_id" : "product_id";
  const editing = record !== null;
  if (isCustomer) editingCustomerId = editing ? record[id] : null;
  else editingProductId = editing ? record[id] : null;
  if (editing) {
    for (const [key, value] of Object.entries(record)) {
      const input = form.elements.namedItem(key);
      if (input) input.value = value ?? "";
    }
  } else {
    form.reset();
    if (!isCustomer) form.elements.stock.value = 0;
  }
  document.querySelector(isCustomer ? "#customer-submit" : "#product-submit").textContent = editing
    ? "Save changes" : `Add ${kind}`;
  document.querySelector(isCustomer ? "#customer-cancel" : "#product-cancel").hidden = !editing;
  form.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function loadCustomers() {
  customers = await request("/customers");
  const table = document.querySelector("#customers-table");
  table.replaceChildren();
  document.querySelector("#customer-count").textContent = customers.length;
  document.querySelector("#customers-empty").hidden = customers.length > 0;
  customers.forEach((customer) => {
    const tr = document.createElement("tr");
    const name = cell(customer.name, "primary-cell");
    const email = cell(customer.email);
    const joined = cell(new Date(customer.created_at).toLocaleDateString());
    const actions = document.createElement("td");
    actions.className = "align-right";
    const group = document.createElement("div");
    group.className = "actions";
    group.append(
      actionButton("Edit", "", () => setEditMode("customer", customer)),
      actionButton("Delete", "delete", async () => {
        if (!confirm(`Delete ${customer.name}? Customers with orders cannot be deleted.`)) return;
        try {
          await request(`/customers/${customer.customer_id}`, { method: "DELETE" });
          if (editingCustomerId === customer.customer_id) setEditMode("customer");
          await refresh();
          showMessage("Customer deleted.");
        } catch (error) { showMessage(error.message, true); }
      }),
    );
    actions.append(group);
    tr.append(name, email, joined, actions);
    table.append(tr);
  });
  const select = document.querySelector("#order-customer");
  const selected = select.value;
  select.replaceChildren(new Option("Choose a customer", ""));
  customers.forEach((c) => select.add(new Option(c.name, c.customer_id)));
  if (customers.some((c) => String(c.customer_id) === selected)) select.value = selected;
}

async function loadProducts() {
  products = await request("/products");
  const table = document.querySelector("#products-table");
  table.replaceChildren();
  document.querySelector("#product-count").textContent = products.length;
  document.querySelector("#products-empty").hidden = products.length > 0;
  products.forEach((product) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.className = "primary-cell";
    name.textContent = product.name;
    if (product.description) {
      const description = document.createElement("span");
      description.className = "secondary-cell";
      description.textContent = product.description;
      name.append(description);
    }
    tr.append(name, cell(`$${Number(product.price).toFixed(2)}`));
    const stock = cell(`${product.stock} in stock`, product.stock < 5 ? "stock-low" : "");
    tr.append(stock);
    const actions = document.createElement("td");
    actions.className = "align-right";
    const group = document.createElement("div");
    group.className = "actions";
    group.append(
      actionButton("Edit", "", () => setEditMode("product", product)),
      actionButton("Delete", "delete", async () => {
        if (!confirm(`Delete ${product.name}? Products in order history cannot be deleted.`)) return;
        try {
          await request(`/products/${product.product_id}`, { method: "DELETE" });
          if (editingProductId === product.product_id) setEditMode("product");
          await refresh();
          showMessage("Product deleted.");
        } catch (error) { showMessage(error.message, true); }
      }),
    );
    actions.append(group);
    tr.append(actions);
    table.append(tr);
  });
  const select = document.querySelector("#order-product");
  const selected = select.value;
  select.replaceChildren(new Option("Choose a product", ""));
  products.forEach((p) => select.add(new Option(`${p.name} · ${p.stock} available`, p.product_id)));
  if (products.some((p) => String(p.product_id) === selected)) select.value = selected;
}

async function loadOrders() {
  const orders = await request("/orders");
  const table = document.querySelector("#orders-table");
  table.replaceChildren();
  document.querySelector("#order-count").textContent = orders.length;
  document.querySelector("#order-value").textContent = `$${orders.reduce((sum, order) => sum + Number(order.total), 0).toFixed(2)}`;
  document.querySelector("#orders-empty").hidden = orders.length > 0;
  orders.forEach((order) => {
    const tr = document.createElement("tr");
    tr.append(cell(`#${order.order_id}`, "primary-cell"), cell(order.customer_name));
    tr.append(cell(new Date(order.created_at).toLocaleDateString()));
    tr.append(cell(`$${Number(order.total).toFixed(2)}`, "primary-cell"));
    const statusCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `badge ${order.status}`;
    badge.textContent = order.status;
    statusCell.append(badge);
    tr.append(statusCell);
    const actions = document.createElement("td");
    actions.className = "align-right";
    const group = document.createElement("div");
    group.className = "actions";
    const status = document.createElement("select");
    status.setAttribute("aria-label", `Status for order ${order.order_id}`);
    ["pending", "paid", "shipped", "cancelled"].forEach((value) => status.add(new Option(value, value)));
    status.value = order.status;
    group.append(status, actionButton("Save", "", async () => {
      try {
        await request(`/orders/${order.order_id}/status`, {
          method: "PATCH", body: JSON.stringify({ status: status.value }),
        });
        await refresh();
        showMessage(`Order #${order.order_id} updated.`);
      } catch (error) { showMessage(error.message, true); }
    }));
    group.append(actionButton("Delete", "delete", async () => {
      if (!confirm(`Delete order #${order.order_id}? Its order lines will also be deleted.`)) return;
      try {
        await request(`/orders/${order.order_id}`, { method: "DELETE" });
        await refresh();
        showMessage(`Order #${order.order_id} deleted.`);
      } catch (error) { showMessage(error.message, true); }
    }));
    actions.append(group);
    tr.append(actions);
    table.append(tr);
  });
}

async function refresh() {
  await Promise.all([loadCustomers(), loadProducts(), loadOrders()]);
}

document.querySelector("#customer-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = Object.fromEntries(new FormData(form));
  try {
    const editing = editingCustomerId !== null;
    await request(editing ? `/customers/${editingCustomerId}` : "/customers", {
      method: editing ? "PUT" : "POST", body: JSON.stringify(data),
    });
    setEditMode("customer");
    await refresh();
    showMessage(editing ? "Customer changes saved." : "Customer added.");
  } catch (error) { showMessage(error.message, true); }
});

document.querySelector("#product-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = Object.fromEntries(new FormData(form));
  data.description = data.description.trim() || null;
  data.price = Number(data.price);
  data.stock = Number(data.stock);
  try {
    const editing = editingProductId !== null;
    await request(editing ? `/products/${editingProductId}` : "/products", {
      method: editing ? "PUT" : "POST", body: JSON.stringify(data),
    });
    setEditMode("product");
    await refresh();
    showMessage(editing ? "Product changes saved." : "Product added.");
  } catch (error) { showMessage(error.message, true); }
});

document.querySelector("#customer-cancel").addEventListener("click", () => setEditMode("customer"));
document.querySelector("#product-cancel").addEventListener("click", () => setEditMode("product"));

document.querySelector("#order-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = Object.fromEntries(new FormData(form));
  try {
    await request("/orders", {
      method: "POST",
      body: JSON.stringify({
        customer_id: Number(data.customer_id),
        items: [{ product_id: Number(data.product_id), quantity: Number(data.quantity) }],
      }),
    });
    form.reset();
    await refresh();
    showMessage("Order created and stock updated.");
  } catch (error) { showMessage(error.message, true); }
});

refresh().catch((error) => showMessage(error.message, true));
```

### Step 4: Make Sure the Backend Is Running

Choose the matching option:

- **Direct run:** Keep the PostgreSQL container and local Uvicorn terminal running.
- **Docker Compose:** From the project root, run:

```bash
docker compose up -d --build
docker compose ps
```

Use only one backend option at a time because both use port `8000`.

### Step 5: Serve the Frontend

From the project root:

```bash
python -m http.server 5500 --directory frontend
```

### Step 6: Open the App

```text
http://localhost:5500
```

### Step 7: Use the App and Inspect PostgreSQL in DBeaver

1. Add a customer and a product in the app.
2. Run `SELECT * FROM customers;` and `SELECT * FROM products;` in DBeaver.
3. Create an order in the app.
4. Query `orders` and `order_items` in DBeaver.
5. Edit customer/product details in the app and repeat the SELECT queries.
6. Delete an unused record in the app and confirm the row is gone in DBeaver.
7. Try deleting a customer or product referenced by order history.

### Step 8: Stop the App

```bash
Ctrl+C
docker compose down
```

