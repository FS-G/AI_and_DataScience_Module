### Backend Development Lectures – Modules 4–6

## Module 4: Crafting Robust, Secure, and Async APIs (Complete Minimal App)

In this module we build a minimal, production-style, fully async FastAPI app that includes:
- Input validation with Pydantic
- Config and secrets via environment variables
- JWT authentication (stateless) with OAuth2 password flow
- Consistent error handling
- Async SQLAlchemy with PostgreSQL
- A tiny protected resource to tie it all together

We’ll implement a “Notes” API: users can register/login and manage their own notes.

### 1) Dependencies
```bash
pip install fastapi "uvicorn[standard]" "sqlalchemy>=2.0" asyncpg pydantic pydantic-settings bcrypt PyJWT python-multipart
```

### 2) Settings and Config
```python
# core/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str  # e.g. postgresql+asyncpg://USER:PASS@HOST:5432/DB
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
```

```
# .env (local only, set vars in Railway for prod)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/notesdb
JWT_SECRET=supersecret
```

### 3) Async Database and Models
```python
# core/db.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from core.settings import settings

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

```python
# models.py
from sqlalchemy import String, Integer, Boolean, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.db import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[list["Note"]] = relationship(back_populates="owner", cascade="all, delete-orphan")

class Note(Base):
    __tablename__ = "notes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    owner: Mapped[User] = relationship(back_populates="notes")
```

### 4) Schemas (Validation)
```python
# schemas.py
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class NoteCreate(BaseModel):
    title: str
    content: str

class NoteRead(BaseModel):
    id: int
    title: str
    content: str
    class Config:
        from_attributes = True
```

### 5) Auth Utilities (JWT + Passwords)
```python
# core/auth.py
from datetime import datetime, timedelta
import bcrypt
import jwt
from core.settings import settings

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expires_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def decode_access_token(token: str) -> str:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return payload.get("sub")
```

### 6) Error Handling
```python
# core/errors.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class DomainError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

def install_error_handlers(app: FastAPI):
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
```

### 7) Routers (Async) — Auth and Notes
```python
# routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.db import get_db
from core.auth import create_access_token, hash_password, verify_password, decode_access_token
from models import User
from schemas import UserCreate, Token

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(400, detail="Email already registered")
    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    await db.commit()
    token = create_access_token(subject=user.email)
    return {"access_token": token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.email == form.username))
    user = res.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(400, detail="Invalid credentials")
    token = create_access_token(subject=user.email)
    return {"access_token": token, "token_type": "bearer"}

async def get_current_user_email(token: str = Depends(oauth2_scheme)) -> str:
    try:
        return decode_access_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
```

```python
# routers/notes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.db import get_db
from models import User, Note
from schemas import NoteCreate, NoteRead
from .auth import get_current_user_email

router = APIRouter(prefix="/notes", tags=["notes"])

@router.post("", response_model=NoteRead)
async def create_note(payload: NoteCreate, email: str = Depends(get_current_user_email), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.email == email))
    user = res.scalar_one()
    note = Note(title=payload.title, content=payload.content, owner_id=user.id)
    db.add(note)
    await db.commit()
    await db.refresh(note)
    return note

@router.get("", response_model=list[NoteRead])
async def list_notes(email: str = Depends(get_current_user_email), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.email == email))
    user = res.scalar_one()
    res2 = await db.execute(select(Note).where(Note.owner_id == user.id))
    notes = res2.scalars().all()
    return notes
```

### 8) Main Application (Async)
```python
# main_async.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from core.db import engine, Base
from core.errors import install_error_handlers
from routers import auth, notes

app = FastAPI(title="Async Notes API")

# CORS: allow browser clients (preflight OPTIONS will return 200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # in prod, set your exact frontend origin
    allow_credentials=True,
    allow_methods=["*"],          # includes OPTIONS for preflight
    allow_headers=["*"],          # includes Content-Type, Authorization
)
install_error_handlers(app)

@app.on_event("startup")
async def on_startup():
    # Create tables on startup (use Alembic in real projects)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

app.include_router(auth.router)
app.include_router(notes.router)

@app.get("/health")
async def health():
    return {"ok": True}
```

Run locally:
```bash
uvicorn main_async:app --reload
```

This complete minimal app demonstrates validation, secrets management, JWT auth, error handling, and async DB access — all in a compact, real-world structure.

### Minimal Frontend (index.html) for the Async Notes App

Use this single HTML file to sign up, log in, create notes, and list notes. Update `API_BASE_URL` to your local or deployed URL.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Async Notes App</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; background:#f7f7fb; margin:0; padding:24px; }
    .card { background:#fff; border:1px solid #e6e6ef; border-radius:8px; padding:16px; margin:0 auto 16px; max-width:720px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
    h1 { text-align:center; margin:0 0 16px; color:#2e2e43; }
    h2 { margin:0 0 12px; color:#3b3b58; font-size:1.1rem; }
    label { display:block; margin:8px 0 4px; color:#4b4b6a; }
    input, textarea { width:100%; padding:10px; border:1px solid #d0d0e0; border-radius:6px; font-size:14px; }
    textarea { resize:vertical; min-height:80px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .btn { background:#4f46e5; color:white; border:none; padding:10px 14px; border-radius:6px; cursor:pointer; margin-top:10px; }
    .btn.secondary { background:#64748b; }
    .muted { color:#6b7280; font-size:13px; margin-top:6px; }
    .list { display:grid; gap:10px; margin-top:10px; }
    .note { border:1px solid #e6e6ef; padding:10px; border-radius:6px; background:#fafafe; }
    .status { color:#10b981; font-weight:600; font-size:13px; }
    .error { background:#fee2e2; color:#991b1b; border-left:4px solid #ef4444; padding:10px; border-radius:6px; margin-top:10px; }
    .success { background:#dcfce7; color:#14532d; border-left:4px solid #22c55e; padding:10px; border-radius:6px; margin-top:10px; }
  </style>
}</head>
<body>
  <h1>Async Notes</h1>

  <div class="card" id="auth-card">
    <h2>Authentication</h2>
    <div id="auth-msg"></div>
    <div class="row">
      <div>
        <label for="reg-email">Register Email</label>
        <input id="reg-email" type="email" placeholder="user@example.com" />
      </div>
      <div>
        <label for="reg-password">Password</label>
        <input id="reg-password" type="password" />
      </div>
    </div>
    <button class="btn" id="register-btn">Register</button>
    <div class="row" style="margin-top:12px;">
      <div>
        <label for="login-email">Login Email</label>
        <input id="login-email" type="email" placeholder="user@example.com" />
      </div>
      <div>
        <label for="login-password">Password</label>
        <input id="login-password" type="password" />
      </div>
    </div>
    <button class="btn" id="login-btn">Login</button>
    <button class="btn secondary" id="logout-btn">Logout</button>
    <div class="muted" id="auth-status">Not authenticated</div>
  </div>

  <div class="card">
    <h2>Create Note</h2>
    <div id="note-msg"></div>
    <label for="note-title">Title</label>
    <input id="note-title" type="text" placeholder="Note title" />
    <label for="note-content">Content</label>
    <textarea id="note-content" placeholder="Write something..."></textarea>
    <button class="btn" id="create-note-btn">Create Note</button>
  </div>

  <div class="card">
    <h2>Your Notes</h2>
    <button class="btn secondary" id="refresh-notes-btn">Refresh</button>
    <div class="list" id="notes-list"></div>
  </div>

  <script>
    // Change this for your environment
    const API_BASE_URL = 'http://localhost:8000';

    function setStatus(elId, msg, kind='success') {
      const el = document.getElementById(elId);
      el.innerHTML = msg ? `<div class="${kind}">${msg}</div>` : '';
      if (msg) setTimeout(() => el.innerHTML = '', 4000);
    }

    function authHeaders() {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = `Bearer ${token}`;
      return headers;
    }

    function updateAuthStatus() {
      const token = localStorage.getItem('token');
      document.getElementById('auth-status').textContent = token ? 'Authenticated' : 'Not authenticated';
    }

    async function register() {
      try {
        const email = document.getElementById('reg-email').value;
        const password = document.getElementById('reg-password').value;
        const resp = await fetch(`${API_BASE_URL}/auth/register`, {
          method: 'POST', headers: authHeaders(), body: JSON.stringify({ email, password })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Register failed');
        localStorage.setItem('token', data.access_token);
        setStatus('auth-msg', 'Registered and logged in', 'success');
        updateAuthStatus();
      } catch (e) {
        setStatus('auth-msg', e.message, 'error');
      }
    }

    async function login() {
      try {
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
        const form = new URLSearchParams();
        form.append('username', email);
        form.append('password', password);
        const resp = await fetch(`${API_BASE_URL}/auth/login`, {
          method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: form
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Login failed');
        localStorage.setItem('token', data.access_token);
        setStatus('auth-msg', 'Logged in', 'success');
        updateAuthStatus();
      } catch (e) {
        setStatus('auth-msg', e.message, 'error');
      }
    }

    function logout() {
      localStorage.removeItem('token');
      updateAuthStatus();
      setStatus('auth-msg', 'Logged out', 'success');
    }

    async function createNote() {
      try {
        const title = document.getElementById('note-title').value;
        const content = document.getElementById('note-content').value;
        const resp = await fetch(`${API_BASE_URL}/notes`, {
          method: 'POST', headers: authHeaders(), body: JSON.stringify({ title, content })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Create failed');
        setStatus('note-msg', 'Note created', 'success');
        document.getElementById('note-title').value = '';
        document.getElementById('note-content').value = '';
        await listNotes();
      } catch (e) {
        setStatus('note-msg', e.message, 'error');
      }
    }

    async function listNotes() {
      try {
        const resp = await fetch(`${API_BASE_URL}/notes`, { headers: authHeaders() });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Fetch failed');
        const container = document.getElementById('notes-list');
        container.innerHTML = data.length ? '' : '<div class="muted">No notes yet</div>';
        for (const n of data) {
          const div = document.createElement('div');
          div.className = 'note';
          div.innerHTML = `<strong>${n.title}</strong><div class="muted">${n.content}</div>`;
          container.appendChild(div);
        }
      } catch (e) {
        const container = document.getElementById('notes-list');
        container.innerHTML = `<div class="error">${e.message}</div>`;
      }
    }

    // Wire events
    document.getElementById('register-btn').addEventListener('click', register);
    document.getElementById('login-btn').addEventListener('click', login);
    document.getElementById('logout-btn').addEventListener('click', logout);
    document.getElementById('create-note-btn').addEventListener('click', createNote);
    document.getElementById('refresh-notes-btn').addEventListener('click', listNotes);

    // On load
    updateAuthStatus();
    listNotes();
  </script>
</body>
</html>
```

---

## Module 5: Quality, Testing, and Advanced Topics

### 1) Software Architecture & Project Structure
Refactor into layers: `routers/`, `schemas/`, `models/`, `services/`, `core/`.

```bash
.
├─ app/
│  ├─ main.py
│  ├─ core/ (config, security, errors)
│  ├─ models/
│  ├─ schemas/
│  ├─ services/
│  └─ routers/
└─ tests/
```

Example service function:
```python
# services/products_service.py
from sqlalchemy.orm import Session
from models import Product
from schemas import ProductCreate

def create_product(db: Session, payload: ProductCreate) -> Product:
    if db.query(Product).filter(Product.name == payload.name).first():
        raise ValueError("Product name exists")
    p = Product(**payload.dict())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p
```

### 2) API Testing Strategies
- Unit tests: test services in isolation.
- Integration tests: spin up app with a test DB (separate schema/database) or mock DB.

Install testing tools:
```bash
pip install pytest httpx
```

Code example with FastAPI TestClient:
```python
# tests/test_products.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_create_and_list_products(monkeypatch):
    # In real integration tests, point DATABASE_URL to a test DB.
    # Here we assume an in-memory or dedicated test environment.
    payload = {"name": "TestItem", "price": "9.99", "quantity": 3}
    r = client.post("/products", json=payload)
    assert r.status_code in (200, 201)
    r2 = client.get("/products")
    assert r2.status_code == 200
    assert any(p["name"] == "TestItem" for p in r2.json())
```

Run tests:
```bash
pytest -q
```

### 3) Real-Time Communication with WebSockets
Use WebSockets for push-style, bidirectional communication.

```python
# websocket_example.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/notify")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass
```

Client test (browser console):
```javascript
const ws = new WebSocket("wss://YOUR-APP.onrailway.app/ws/notify");
ws.onmessage = (e) => console.log(e.data);
ws.onopen = () => ws.send("hello");
```

---

## Module 6: Final Capstone Project & Continuous Deployment on Railway

### 1) Project Scoping and Design
Build a simple blogging platform with users, posts, and comments.
- Users: register/login (JWT), list profile
- Posts: CRUD; owner = user; pagination
- Comments: CRUD; linked to posts and users

### 2) Full Application Implementation & Incremental Deployment
Workflow:
- Implement feature → push to GitHub → Railway deploys automatically
- Maintain DB migrations with Alembic for each schema change
- Use feature branches and PRs for reviews (optional)

### 3) What the Final Codebase Demonstrates
- **Authentication**: JWT-based login and protected routes
- **Relational Modeling**: Users–Posts–Comments with foreign keys
- **Validation & Errors**: Pydantic models and custom handlers
- **Structure**: Routers, services, schemas, models, core modules
- **CD**: GitHub → Railway push-to-deploy pipeline

### 4) Continuous Deployment Setup on Railway
- Connect GitHub repo to Railway service.
- Configure start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Environment variables: `DATABASE_URL`, `JWT_SECRET`, etc.
- Create migrations per feature:
```bash
alembic revision --autogenerate -m "add comments"
alembic upgrade head
```

### 5) Production Monitoring (Basics)
In Railway UI:
- **Logs**: View live application logs for errors and requests.
- **Metrics**: Check CPU/memory usage to right-size your service.
- **Env Vars**: Rotate secrets safely; no code changes required.

Practical:
- Trigger a deployment and watch logs during rollout.
- Validate health endpoints and DB connectivity post-deploy.

This final capstone consolidates all modules into a continuously deployed, monitored API on Railway, ready for real use and extension.



---
