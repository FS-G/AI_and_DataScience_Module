# Docker & Containerization

## Why Do We Need Docker?

### The Problem
- **"It works on my machine"** - Different environments (dev, staging, production) cause issues
- Dependency conflicts between applications
- Complex setup for new team members
- Different OS requirements
- Version mismatches (Python 3.8 vs 3.11, different libraries)

### The Solution: Containers
- Package application + dependencies + environment together
- Run anywhere consistently
- Isolated from other applications
- Lightweight compared to Virtual Machines

### Docker vs Virtual Machines
```
VM: OS → Hypervisor → [Guest OS + App] [Guest OS + App]
Docker: OS → Docker Engine → [App] [App] (shares host OS kernel)
```

**Benefits:**
- Faster startup (seconds vs minutes)
- Less resource usage
- Easier to scale
- Version control for entire environments

---

## Installation

### Linux
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### Windows/Mac
Download Docker Desktop from docker.com

### Verify Installation
```bash
docker --version
docker run hello-world
```

---

## Core Concepts

### 1. Images
A blueprint/template for containers. Read-only.

```bash
# List images
docker images

# Pull an image
docker pull python:3.11

# Remove an image
docker rmi image_name
```

### 2. Containers
Running instance of an image. Can be started, stopped, deleted.

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Run a container
docker run -d --name myapp python:3.11

# Stop container
docker stop myapp

# Remove container
docker rm myapp
```

### 3. Dockerfile
Instructions to build an image.

### 4. Docker Hub
Registry to store and share images (like GitHub for Docker).

---

## Building a FastAPI Application with Docker

### Step 1: Create FastAPI App

**app.py**
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from Docker!"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "name": f"Item {item_id}"}
```

**requirements.txt**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
```

### Step 2: Create Dockerfile

**Dockerfile**
```dockerfile
# Base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 3: Build the Image

```bash
docker build -t fastapi-app .
```

**Breakdown:**
- `docker build` - Build an image
- `-t fastapi-app` - Tag/name the image
- `.` - Build context (current directory)

### Step 4: Run the Container

```bash
docker run -d -p 8000:8000 --name my-fastapi fastapi-app
```

**Breakdown:**
- `-d` - Detached mode (run in background)
- `-p 8000:8000` - Port mapping (host:container)
- `--name my-fastapi` - Container name
- `fastapi-app` - Image to use

### Step 5: Test It

```bash
curl http://localhost:8000
# {"message": "Hello from Docker!"}
```

---

## Essential Docker Commands

### Container Management
```bash
# Run container interactively
docker run -it python:3.11 bash

# Execute command in running container
docker exec -it my-fastapi bash

# View logs
docker logs my-fastapi
docker logs -f my-fastapi  # Follow logs

# Restart container
docker restart my-fastapi

# Stop all containers
docker stop $(docker ps -q)

# Remove all stopped containers
docker container prune
```

### Image Management
```bash
# Build with no cache
docker build --no-cache -t fastapi-app .

# Tag an image
docker tag fastapi-app myusername/fastapi-app:v1

# View image history
docker history fastapi-app

# Remove unused images
docker image prune
```

---

## Docker Hub

Docker Hub is a cloud registry service for storing and sharing Docker images (like GitHub for Docker).

### Push to Docker Hub

**Step 1: Create Docker Hub Account**
- Go to https://hub.docker.com
- Sign up and note your username (e.g., if your username is `john123`, this is what you'll use)

**Step 2: Create a Repository**
- Login to Docker Hub
- Click "Create Repository"
- Name it (e.g., `fastapi-app`)
- Choose Public or Private
- Click "Create"

**Step 3: Login via CLI**
```bash
# Login (enter your Docker Hub username and password)
docker login
```

**Step 4: Tag Your Image**
```bash
# Format: docker tag local-image your-dockerhub-username/repo-name:tag
docker tag fastapi-app john123/fastapi-app:latest

# Example with version tag
docker tag fastapi-app john123/fastapi-app:v1.0
```

**Note:** Replace `john123` with YOUR actual Docker Hub username!

**Step 5: Push to Docker Hub**
```bash
# Push to your repository
docker push john123/fastapi-app:latest
```

### Pull from Docker Hub

Now anyone can pull and run your image:

```bash
# Pull the image
docker pull john123/fastapi-app:latest

# Run it
docker run -p 8000:8000 john123/fastapi-app:latest
```

**Public vs Private Repositories:**
- **Public** - Anyone can pull (like open source)
- **Private** - Only you and authorized users can access

---

## Volumes (Data Persistence)

Containers are ephemeral - data is lost when removed. Volumes persist data.

```bash
# Create volume
docker volume create mydata

# Run with volume (includes port mapping!)
docker run -d -p 8000:8000 -v mydata:/app/data fastapi-app

# Mount local directory (includes port mapping!)
docker run -d -p 8000:8000 -v $(pwd)/data:/app/data fastapi-app

# List volumes
docker volume ls
```

**Example: Database with persistence**
```bash
docker run -d --name postgres -e POSTGRES_PASSWORD=secret -v pgdata:/var/lib/postgresql/data -p 5432:5432 postgres:15
```

---

## Networks

Containers can communicate via Docker networks.

```bash
# Create network
docker network create mynetwork

# Run containers on same network
docker run -d --name api --network mynetwork fastapi-app
docker run -d --name db --network mynetwork postgres:15

# Now 'api' can connect to 'db' using hostname 'db'
```

---

## Docker Compose

Manage multi-container applications with a single file.

**Setup:**
1. Create `docker-compose.yml` in your FastAPI project folder (same folder as `Dockerfile` and `app.py`)
2. Open terminal in that same folder
3. Run the commands below

**docker-compose.yml**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      - db
    volumes:
      - ./app.py:/app/app.py
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### Docker Compose Commands

**Important:** Run these commands in the same folder where `docker-compose.yml` is located!

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Scale services
docker-compose up -d --scale api=3
```

---

## Environment Variables

```bash
# Pass at runtime
docker run -e API_KEY=secret123 fastapi-app

# Using .env file
docker run --env-file .env fastapi-app
```

**In Dockerfile:**
```dockerfile
ENV APP_ENV=production
ENV PORT=8000
```

---

## Multi-Stage Builds

Reduce image size by using multiple stages.

```dockerfile
# Build stage
FROM python:3.11 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app.py .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

---

## .dockerignore

Exclude files from build context (faster builds).

**.dockerignore**
```
__pycache__
*.pyc
.git
.env
node_modules
*.md
```

---

## Best Practices

1. **Use specific image tags** - `python:3.11-slim` not `python:latest`
2. **Minimize layers** - Combine RUN commands
3. **Order matters** - Copy dependencies before code (better caching)
4. **Use .dockerignore** - Exclude unnecessary files
5. **Run as non-root user**
```dockerfile
RUN useradd -m appuser
USER appuser
```
6. **Keep images small** - Use slim/alpine variants
7. **One process per container** - Don't run multiple services
8. **Use health checks**
```dockerfile
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/ || exit 1
```

---

## Common Dockerfile Instructions

```dockerfile
FROM python:3.11-slim       # Base image
WORKDIR /app                # Set working directory
COPY . .                    # Copy files
ADD file.tar.gz /app        # Copy + extract archives
RUN pip install -r req.txt  # Execute command during build
CMD ["python", "app.py"]    # Default command (can override)
ENTRYPOINT ["python"]       # Fixed command (append args)
EXPOSE 8000                 # Document port (doesn't publish)
ENV KEY=value               # Environment variable
VOLUME /data                # Mount point
USER appuser                # Switch user
ARG BUILD_VERSION           # Build-time variable
LABEL version="1.0"         # Metadata
```

---

## Debugging

```bash
# Interactive shell in container
docker exec -it container_name bash

# View container details
docker inspect container_name

# View resource usage
docker stats

# Copy files from container
docker cp container_name:/app/file.txt ./

# View changes to container filesystem
docker diff container_name
```

---

## Clean Up

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a --volumes
```

---

## Summary

**Workflow:**
1. Write application code
2. Create Dockerfile
3. Build image: `docker build -t myapp .`
4. Run container: `docker run -p 8000:8000 myapp`
5. Push to registry: `docker push myapp`
6. Deploy anywhere Docker runs

**Key Concepts:**
- **Image** = Blueprint (immutable)
- **Container** = Running instance
- **Volume** = Persistent data
- **Network** = Container communication
- **Compose** = Multi-container orchestration

**Why Docker?**
- Consistency across environments
- Easy deployment
- Isolation
- Scalability
- Reproducibility