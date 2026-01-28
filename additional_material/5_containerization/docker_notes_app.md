# Docker Full-Stack Notes App
## Real-World Development Workflow

This guide builds a complete notes application with:
- **Backend:** FastAPI + PostgreSQL
- **Frontend:** HTML/CSS/JS
- **Docker:** Multi-container setup with volumes, networks, and Docker Hub

---

## Project Structure

```
notes-app/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   └── Dockerfile
├── docker-compose.yml
└── .env
```

---

## Step 1: Backend (FastAPI + PostgreSQL)

### backend/app.py

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import os

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        database=os.getenv("DB_NAME", "notesdb"),
        user=os.getenv("DB_USER", "admin"),
        password=os.getenv("DB_PASSWORD", "secret123"),
        cursor_factory=RealDictCursor
    )
    return conn

# Initialize database
@app.on_event("startup")
def startup():
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(64) NOT NULL
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            title VARCHAR(100) NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()

# Models
class User(BaseModel):
    username: str
    password: str

class Note(BaseModel):
    title: str
    content: str

class NoteWithUser(Note):
    user_id: int

# Hash password
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Routes
@app.get("/")
def root():
    return {"message": "Notes API is running"}

@app.post("/signup")
def signup(user: User):
    conn = get_db()
    cur = conn.cursor()
    
    try:
        hashed_pw = hash_password(user.password)
        cur.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id",
            (user.username, hashed_pw)
        )
        user_id = cur.fetchone()["id"]
        conn.commit()
        return {"message": "User created", "user_id": user_id}
    except psycopg2.IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        cur.close()
        conn.close()

@app.post("/login")
def login(user: User):
    conn = get_db()
    cur = conn.cursor()
    
    hashed_pw = hash_password(user.password)
    cur.execute(
        "SELECT id, username FROM users WHERE username=%s AND password=%s",
        (user.username, hashed_pw)
    )
    result = cur.fetchone()
    
    cur.close()
    conn.close()
    
    if result:
        return {"message": "Login successful", "user_id": result["id"], "username": result["username"]}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/notes/{user_id}")
def get_notes(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute(
        "SELECT id, title, content, created_at FROM notes WHERE user_id=%s ORDER BY created_at DESC",
        (user_id,)
    )
    notes = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return {"notes": notes}

@app.post("/notes")
def create_note(note: NoteWithUser):
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute(
        "INSERT INTO notes (user_id, title, content) VALUES (%s, %s, %s) RETURNING id",
        (note.user_id, note.title, note.content)
    )
    note_id = cur.fetchone()["id"]
    conn.commit()
    
    cur.close()
    conn.close()
    
    return {"message": "Note created", "note_id": note_id}

@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("DELETE FROM notes WHERE id=%s", (note_id,))
    conn.commit()
    
    cur.close()
    conn.close()
    
    return {"message": "Note deleted"}
```

### backend/requirements.txt

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
psycopg2-binary==2.9.9
pydantic==2.5.0
```

### backend/Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 2: Frontend (HTML/CSS/JS)

### frontend/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Notes App</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .auth-section, .notes-section { background: white; padding: 30px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2 { margin-bottom: 20px; color: #333; }
        input, textarea { width: 100%; padding: 12px; margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; margin-right: 8px; }
        button:hover { background: #0056b3; }
        button.delete { background: #dc3545; }
        button.delete:hover { background: #c82333; }
        .note { background: #fff9e6; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #ffd700; }
        .note h3 { margin-bottom: 8px; color: #333; }
        .note p { color: #666; margin-bottom: 8px; }
        .note small { color: #999; }
        .hidden { display: none; }
        .user-info { background: #e7f3ff; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
        .logout { background: #6c757d; }
        .logout:hover { background: #545b62; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 Notes App</h1>
        
        <!-- Auth Section -->
        <div id="authSection" class="auth-section">
            <h2>Login / Signup</h2>
            <input type="text" id="username" placeholder="Username">
            <input type="password" id="password" placeholder="Password">
            <button onclick="login()">Login</button>
            <button onclick="signup()">Signup</button>
            <p id="authMessage"></p>
        </div>

        <!-- Notes Section -->
        <div id="notesSection" class="notes-section hidden">
            <div class="user-info">
                <span>Logged in as: <strong id="currentUser"></strong></span>
                <button class="logout" onclick="logout()">Logout</button>
            </div>
            
            <h2>Create Note</h2>
            <input type="text" id="noteTitle" placeholder="Note Title">
            <textarea id="noteContent" placeholder="Note Content" rows="4"></textarea>
            <button onclick="createNote()">Add Note</button>
            
            <h2>My Notes</h2>
            <div id="notesList"></div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:8000';
        let currentUserId = null;
        let currentUsername = null;

        async function signup() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch(`${API_URL}/signup`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('authMessage').textContent = 'Signup successful! Now login.';
                    document.getElementById('authMessage').style.color = 'green';
                } else {
                    document.getElementById('authMessage').textContent = data.detail;
                    document.getElementById('authMessage').style.color = 'red';
                }
            } catch (error) {
                document.getElementById('authMessage').textContent = 'Error: ' + error.message;
                document.getElementById('authMessage').style.color = 'red';
            }
        }

        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch(`${API_URL}/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentUserId = data.user_id;
                    currentUsername = data.username;
                    document.getElementById('currentUser').textContent = currentUsername;
                    document.getElementById('authSection').classList.add('hidden');
                    document.getElementById('notesSection').classList.remove('hidden');
                    loadNotes();
                } else {
                    document.getElementById('authMessage').textContent = data.detail;
                    document.getElementById('authMessage').style.color = 'red';
                }
            } catch (error) {
                document.getElementById('authMessage').textContent = 'Error: ' + error.message;
                document.getElementById('authMessage').style.color = 'red';
            }
        }

        async function loadNotes() {
            try {
                const response = await fetch(`${API_URL}/notes/${currentUserId}`);
                const data = await response.json();
                
                const notesList = document.getElementById('notesList');
                notesList.innerHTML = '';
                
                data.notes.forEach(note => {
                    const noteDiv = document.createElement('div');
                    noteDiv.className = 'note';
                    noteDiv.innerHTML = `
                        <h3>${note.title}</h3>
                        <p>${note.content}</p>
                        <small>${new Date(note.created_at).toLocaleString()}</small>
                        <button class="delete" onclick="deleteNote(${note.id})">Delete</button>
                    `;
                    notesList.appendChild(noteDiv);
                });
            } catch (error) {
                console.error('Error loading notes:', error);
            }
        }

        async function createNote() {
            const title = document.getElementById('noteTitle').value;
            const content = document.getElementById('noteContent').value;
            
            if (!title || !content) {
                alert('Please fill in both title and content');
                return;
            }
            
            try {
                const response = await fetch(`${API_URL}/notes`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: currentUserId, title, content })
                });
                
                if (response.ok) {
                    document.getElementById('noteTitle').value = '';
                    document.getElementById('noteContent').value = '';
                    loadNotes();
                }
            } catch (error) {
                console.error('Error creating note:', error);
            }
        }

        async function deleteNote(noteId) {
            if (!confirm('Delete this note?')) return;
            
            try {
                await fetch(`${API_URL}/notes/${noteId}`, { method: 'DELETE' });
                loadNotes();
            } catch (error) {
                console.error('Error deleting note:', error);
            }
        }

        function logout() {
            currentUserId = null;
            currentUsername = null;
            document.getElementById('authSection').classList.remove('hidden');
            document.getElementById('notesSection').classList.add('hidden');
            document.getElementById('username').value = '';
            document.getElementById('password').value = '';
            document.getElementById('authMessage').textContent = '';
        }
    </script>
</body>
</html>
```

### frontend/Dockerfile

```dockerfile
FROM nginx:alpine

# Copy HTML file to nginx default directory
COPY index.html /usr/share/nginx/html/

# Expose port 80
EXPOSE 80

# Nginx runs automatically
```

---

## Step 3: Docker Compose Setup

### docker-compose.yml

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  db:
    image: postgres:15
    container_name: notes_db
    environment:
      - POSTGRES_DB=notesdb
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=secret123
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - notes_network
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d notesdb"]
      interval: 10s
      timeout: 5s
      retries: 5

  # FastAPI Backend
  backend:
    build: ./backend
    container_name: notes_backend
    environment:
      - DB_HOST=db
      - DB_NAME=notesdb
      - DB_USER=admin
      - DB_PASSWORD=secret123
    ports:
      - "8000:8000"
    networks:
      - notes_network
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  # Frontend (Nginx)
  frontend:
    build: ./frontend
    container_name: notes_frontend
    ports:
      - "3000:80"
    networks:
      - notes_network
    depends_on:
      - backend
    restart: unless-stopped

# Named volume for database persistence
volumes:
  postgres_data:

# Custom network for container communication
networks:
  notes_network:
    driver: bridge
```

---

## Step 4: Real-World Development Workflow

### Phase 1: Local Development

**1. Create project structure**
```bash
mkdir notes-app
cd notes-app
mkdir backend frontend
```

**2. Add all files** (app.py, requirements.txt, Dockerfiles, index.html, docker-compose.yml)

**3. Build and run locally**
```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f
```

**4. Test the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Database: localhost:5432

**5. Development tips**
```bash
# Rebuild after code changes
docker-compose up -d --build

# View specific service logs
docker-compose logs backend

# Enter backend container
docker exec -it notes_backend bash

# Enter database
docker exec -it notes_db psql -U admin -d notesdb
```

### Phase 2: Version Control

**Create .dockerignore in backend/**
```
__pycache__
*.pyc
.env
*.log
```

**Create .gitignore**
```
__pycache__/
*.pyc
.env
postgres_data/
```

### Phase 3: Push to Docker Hub

**1. Tag images**
```bash
# Replace 'yourusername' with your Docker Hub username
docker tag notes-app-backend yourusername/notes-backend:v1.0
docker tag notes-app-frontend yourusername/notes-frontend:v1.0
```

**2. Login and push**
```bash
docker login
docker push yourusername/notes-backend:v1.0
docker push yourusername/notes-frontend:v1.0
```

**3. Update docker-compose.yml for production**
```yaml
services:
  backend:
    image: yourusername/notes-backend:v1.0  # Use image instead of build
    # ... rest of config
    
  frontend:
    image: yourusername/notes-frontend:v1.0  # Use image instead of build
    # ... rest of config
```

### Phase 4: Deploy to Production Server

**Push code to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/notes-app.git
git push -u origin main
```

**On production server:**
```bash
# Option 1: Clone from GitHub
git clone https://github.com/yourusername/notes-app.git
cd notes-app

# Option 2: Download just docker-compose.yml (if using pre-built images from Docker Hub)
wget https://raw.githubusercontent.com/yourusername/notes-app/main/docker-compose.yml

# Pull images from Docker Hub
docker-compose pull

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

---

## Key Docker Concepts Demonstrated

### 1. **Volumes** (postgres_data)
- Persists database data even when container is removed
- Data survives container restarts
- Can be backed up separately

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect notes-app_postgres_data

# Backup database
docker exec notes_db pg_dump -U admin notesdb > backup.sql
```

### 2. **Networks** (notes_network)
- Containers communicate using service names (db, backend, frontend)
- Backend connects to database using hostname `db`
- Isolated from other Docker networks

```bash
# Inspect network
docker network inspect notes-app_notes_network

# See which containers are connected
docker network inspect notes-app_notes_network | grep Name
```

### 3. **Multi-Container Architecture**
- Each service runs in its own container
- Services can be scaled independently
- Easier to update individual components

```bash
# Scale backend to 3 instances
docker-compose up -d --scale backend=3
```

### 4. **Environment Variables**
- Sensitive data (passwords) kept in environment
- Easy to change between dev/staging/prod
- Can use .env file

**Create .env file:**
```
DB_PASSWORD=secret123
DB_USER=admin
DB_NAME=notesdb
```

**Update docker-compose.yml:**
```yaml
services:
  db:
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
```

### 5. **Health Checks**
- Backend waits for database to be ready
- Prevents connection errors on startup
- Auto-restart on failures

### 6. **Port Mapping**
```
frontend: 3000:80   → localhost:3000 → container:80
backend:  8000:8000 → localhost:8000 → container:8000
db:       5432:5432 → localhost:5432 → container:5432
```

---

## Useful Commands for This Setup

```bash
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# Stop and remove volumes (DELETES DATA!)
docker-compose down -v

# View logs for all services
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend

# Restart specific service
docker-compose restart backend

# Rebuild specific service
docker-compose up -d --build backend

# Execute command in backend
docker-compose exec backend python -c "print('Hello')"

# Access database
docker-compose exec db psql -U admin -d notesdb

# Check resource usage
docker stats
```

---

## Troubleshooting

### Backend can't connect to database
```bash
# Check if db is running
docker-compose ps

# Check db logs
docker-compose logs db

# Test connection from backend
docker-compose exec backend ping db
```

### Frontend can't reach backend
- Make sure API_URL in index.html is correct
- Check CORS settings in app.py
- Verify backend is running: `curl http://localhost:8000`

### Port already in use
```bash
# Find what's using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8001:8000"
```

---

## Production Best Practices

1. **Use specific image tags**
```yaml
image: postgres:15  # Not 'latest'
```

2. **Use secrets for passwords**
```yaml
secrets:
  db_password:
    file: ./db_password.txt
```

3. **Add restart policies**
```yaml
restart: unless-stopped
```

4. **Use multi-stage builds** (reduces image size)
```dockerfile
FROM python:3.11 AS builder
# ... install dependencies

FROM python:3.11-slim
COPY --from=builder ...
```

5. **Run health checks**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/"]
  interval: 30s
  timeout: 10s
  retries: 3
```

6. **Use environment-specific compose files**
```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

---

## Summary: Real-World Flow

1. **Develop locally** with Docker Compose
2. **Test** all services work together
3. **Build images** for each service
4. **Push to Docker Hub** (or private registry)
5. **Deploy** by pulling images on production server
6. **Scale** services as needed
7. **Monitor** logs and health
8. **Update** by pushing new versions

This setup mirrors professional development workflows used in production environments!