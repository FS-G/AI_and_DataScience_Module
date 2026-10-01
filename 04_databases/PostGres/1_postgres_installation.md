# PostgreSQL, Docker, and DBeaver Installation Guide

This guide runs PostgreSQL in Docker and connects to it with DBeaver. Install Docker for your operating system, then use the shared Compose setup below.

## 1. Install Docker

### Windows

1. Enable hardware virtualization in BIOS/UEFI.
2. Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
3. Use the WSL 2 backend when offered. Follow Docker Desktop's prompt to install or update WSL if needed.
4. Restart Windows if the installer requests it.
5. Start Docker Desktop and wait until it reports that Docker is running.
6. Open PowerShell and verify the installation:

```powershell
docker --version
docker compose version
```

### macOS

1. Check whether your Mac uses Apple silicon or an Intel processor.
2. Download the matching [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) installer.
3. Install Docker Desktop, start it, and wait until it reports that Docker is running.
4. Open Terminal and verify:

```bash
docker --version
docker compose version
```

### Linux

1. Install Docker Engine and the Docker Compose plugin using the instructions for your distribution in the [official Docker Engine guide](https://docs.docker.com/engine/install/).
2. Start the Docker service, if the installation did not start it:

```bash
sudo systemctl enable --now docker
```

3. Verify Docker and Compose:

```bash
docker --version
docker compose version
```

4. If using Docker without `sudo` is configured for your account, sign out and back in after adding your user to the `docker` group. Otherwise, prefix Docker commands with `sudo`.

## 2. Start PostgreSQL in a Docker Container

The remaining commands are the same on Windows, macOS, and Linux. Use PowerShell on Windows, or Terminal on macOS/Linux. On Linux installations where Docker requires elevated privileges, add `sudo` before each `docker` or `docker compose` command.

### Create a project folder

```bash
mkdir postgres-demo
cd postgres-demo
```

Create `compose.yaml` in that folder:

```yaml
services:
  postgres:
    image: postgres:17
    container_name: postgres-demo
    restart: unless-stopped
    environment:
      POSTGRES_DB: shop_db
      POSTGRES_USER: shop_user
      POSTGRES_PASSWORD: shop_pass_123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U shop_user -d shop_db"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  postgres_data:
```

For a classroom demo, these credentials are convenient. Do not reuse the sample password for a public or production database.

### Start and inspect the container

```bash
docker compose up -d
docker compose ps
docker compose logs postgres
```

Wait for the health status to become `healthy`.

### Verify PostgreSQL from inside the container

```bash
docker exec -it postgres-demo psql -U shop_user -d shop_db
```

At the `shop_db=>` prompt, run:

```sql
SELECT version();
\conninfo
\q
```

### Container controls

```bash
docker compose stop
docker compose start
docker compose down
```

`docker compose down` keeps the named `postgres_data` volume. To intentionally remove the database data too, use `docker compose down -v`.

## 3. Install DBeaver Community

1. Download [DBeaver Community](https://dbeaver.io/download/) for Windows, macOS, or Linux.
2. Install it using the standard installer for your operating system. On Linux, choose the package matching your distribution, or use the available archive/package instructions on the download page.
3. Open DBeaver.

## 4. Connect DBeaver to the Container

1. In DBeaver, choose **Database → New Database Connection**.
2. Select **PostgreSQL** and click **Next**.
3. Enter these connection settings:

| Setting | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Database | `shop_db` |
| Username | `shop_user` |
| Password | `shop_pass_123` |

4. Click **Test Connection**.
5. If prompted, allow DBeaver to download the PostgreSQL JDBC driver.
6. Confirm the test succeeds, then click **Finish**.
7. Open an SQL Editor for the connection and run:

```sql
SELECT current_database(), current_user, version();
```

The result should show `shop_db` and `shop_user`.

## Troubleshooting

### Docker command is not found

- **Windows/macOS:** Start Docker Desktop and reopen PowerShell/Terminal after installation.
- **Linux:** Check Docker Engine and the Compose plugin are installed; try prefixing the command with `sudo` if your user lacks Docker permission.

### Port 5432 is already in use

- Stop another local PostgreSQL service, or change the Compose port mapping to `5433:5432`.
- If changed, use port `5433` in DBeaver.

### DBeaver cannot connect

- Check `docker compose ps` and wait for PostgreSQL to be healthy.
- Check the host is `localhost` and the port matches the Compose mapping.
- Check the database, username, and password match `compose.yaml`.
- If you mapped a different host port, enter that port in DBeaver; the container still listens on port `5432`.

### Data disappeared after restarting

- Confirm the Compose file includes `postgres_data` and that you did not run `docker compose down -v`.

## Quick Checklist

- [ ] Docker starts and `docker --version` works.
- [ ] `postgres-demo` is healthy.
- [ ] `psql` connects to `shop_db`.
- [ ] DBeaver connection test succeeds.
- [ ] The verification query runs in DBeaver.

## Remove Everything and Reset for the Next Class

From the folder containing `compose.yaml`, run:

```bash
docker compose down --volumes --remove-orphans
```

This stops and removes the containers and deletes the PostgreSQL data volume. Run `docker compose up -d` again to create a fresh, empty database with the same settings.
