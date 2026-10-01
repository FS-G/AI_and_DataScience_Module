# MySQL and MySQL Workbench Installation Guide (Windows & Linux)

This guide will walk you through downloading and installing MySQL Server and MySQL Workbench on Windows and Linux systems.

## Table of Contents
- [Windows Installation](#windows-installation)
- [Linux Installation](#linux-installation)
- [Initial Setup](#initial-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Windows Installation

### Method 1: MySQL Installer (Recommended)

#### Step 1: Download MySQL Installer

1. Go to [https://dev.mysql.com/downloads/installer/](https://dev.mysql.com/downloads/installer/)
2. You'll see two options:
   - **mysql-installer-web-community-8.x.xx.msi** (smaller, downloads components during installation)
   - **mysql-installer-community-8.x.xx.msi** (larger, contains all components offline)
3. Choose the **larger offline installer** for more reliable installation
4. Click "Download" (you don't need to create an Oracle account - click "No thanks, just start my download")

#### Step 2: Run the MySQL Installer

1. Double-click the downloaded `.msi` file
2. Choose installation type:
   - **"Full"** - Installs everything (MySQL Server, Workbench, connectors, documentation)
   - **"Custom"** - Select only what you need
   - Choose **"Full"** for complete installation
3. Click "Next" → "Execute" to download and install all components
4. Wait for all components to install (this may take several minutes)

#### Step 3: Configure MySQL Server

**⚠️ CRITICAL: ROOT PASSWORD SETUP**

1. After installation, you'll reach "Product Configuration"
2. Click "Next" to configure MySQL Server
3. Choose "Standalone MySQL Server" → "Next"
4. Select "Development Computer" (uses less resources) → "Next"
5. **ROOT PASSWORD SETUP** (Most Important Step):
   - Enter a strong password in "MySQL Root Password"
   - **⚠️ WRITE THIS PASSWORD DOWN AND REMEMBER IT!** 
   - This is the master password for your MySQL installation
   - You'll need this password every time you connect to MySQL
   - Re-enter the same password in "Repeat Password"
6. You can add additional users here, or do it later
7. Click "Next"

#### Step 4: Complete Configuration

1. Choose to run MySQL as a Windows Service (recommended)
2. Keep "Start the MySQL Server at System Startup" checked
3. Click "Execute" to apply all configurations
4. Click "Finish" when configuration is complete
5. MySQL Workbench is already installed and ready to use!

### Method 2: Separate Downloads

#### Download MySQL Server

1. Go to [https://dev.mysql.com/downloads/mysql/](https://dev.mysql.com/downloads/mysql/)
2. Select "Microsoft Windows" from Operating System dropdown
3. Choose "Windows (x86, 64-bit), ZIP Archive" for manual installation
4. Click "Download" → "No thanks, just start my download"

#### Download MySQL Workbench

1. Go to [https://dev.mysql.com/downloads/workbench/](https://dev.mysql.com/downloads/workbench/)
2. Select "Microsoft Windows" from Operating System dropdown
3. Choose "Windows (x86, 64-bit), MSI Installer"
4. Click "Download" → "No thanks, just start my download"

#### Install Both

1. Extract MySQL Server ZIP to a folder (e.g., `C:\mysql`)
2. Run MySQL Workbench MSI installer
3. You'll need to manually configure MySQL Server (advanced users)

---

## Linux Installation

### Ubuntu/Debian Systems

#### Step 1: Update System

```bash
sudo apt update
sudo apt upgrade
```

#### Step 2: Download and Install MySQL Server

**Option A: Using APT Repository**

```bash
# Install MySQL Server
sudo apt install mysql-server

# Start MySQL service
sudo systemctl start mysql
sudo systemctl enable mysql
```

**Option B: Download from MySQL Website**

1. Go to [https://dev.mysql.com/downloads/mysql/](https://dev.mysql.com/downloads/mysql/)
2. Select "Linux - Generic" from Operating System
3. Download the "Linux - Generic (glibc 2.28) (x86, 64-bit), Compressed TAR Archive"
4. Extract and follow manual installation process (advanced)

#### Step 3: Secure MySQL Installation

**⚠️ CRITICAL: ROOT PASSWORD SETUP**

```bash
sudo mysql_secure_installation
```

This will prompt you to:
1. **Set root password** - **⚠️ CHOOSE A STRONG PASSWORD AND REMEMBER IT!**
2. Remove anonymous users (choose Y)
3. Disallow root login remotely (choose Y for security)
4. Remove test database (choose Y)
5. Reload privilege tables (choose Y)

**Write down your root password immediately!**

#### Step 4: Install MySQL Workbench

**Option A: Using APT**

```bash
# Add MySQL APT repository (if not already added)
wget https://dev.mysql.com/get/mysql-apt-config_0.8.29-1_all.deb
sudo dpkg -i mysql-apt-config_0.8.29-1_all.deb
sudo apt update

# Install Workbench
sudo apt install mysql-workbench-community
```

**Option B: Download DEB Package**

1. Go to [https://dev.mysql.com/downloads/workbench/](https://dev.mysql.com/downloads/workbench/)
2. Select "Ubuntu Linux" from Operating System
3. Choose your Ubuntu version
4. Download the `.deb` file
5. Install with: `sudo dpkg -i mysql-workbench-community_*.deb`

### CentOS/RHEL/Fedora Systems

#### Step 1: Install MySQL Server

```bash
# For Fedora/CentOS 8+
sudo dnf install mysql-server

# For older CentOS/RHEL
sudo yum install mysql-community-server

# Start and enable MySQL
sudo systemctl start mysqld
sudo systemctl enable mysqld
```

#### Step 2: Get Initial Root Password

```bash
# Find the temporary password
sudo grep 'temporary password' /var/log/mysqld.log
```

**⚠️ Copy this temporary password - you'll need it!**

#### Step 3: Secure Installation

```bash
mysql_secure_installation
```

1. Enter the temporary password you found above
2. **Set a new strong root password** - **⚠️ REMEMBER THIS PASSWORD!**
3. Answer Y to all security questions

#### Step 4: Install MySQL Workbench

```bash
# Download and install Workbench
sudo dnf install mysql-workbench-community
# or for older systems:
# sudo yum install mysql-workbench-community
```

---

## Initial Setup

### Start MySQL Service

**Windows**: MySQL should start automatically if installed as a service

**Linux**:
```bash
sudo systemctl start mysql
sudo systemctl status mysql  # Check if running
```

### First Connection Test

**Test MySQL Server Connection**:
```bash
mysql -u root -p
```
**Enter the root password you set during installation!**

**Expected Output**:
```
Welcome to the MySQL monitor. Commands end with ; or \g.
Your MySQL connection id is X
Server version: 8.0.xx MySQL Community Server - GPL

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql>
```

If you see this, MySQL Server is working correctly!

### Create a Regular User (Recommended)

```sql
-- Create a new user for daily use (don't use root for everything)
CREATE USER 'myuser'@'localhost' IDENTIFIED BY 'myuserpassword';
GRANT ALL PRIVILEGES ON *.* TO 'myuser'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
EXIT;
```

**⚠️ Remember this user password too!**

---

## Verification

### 1. Test MySQL Workbench Connection

1. **Open MySQL Workbench**
2. You should see a connection named "Local instance MySQL80" or similar
3. **Click on it**
4. **Enter your root password** (the one you set during installation)
5. Click "OK"
6. You should see the Workbench main interface

### 2. Create Test Database

In Workbench, create a new SQL tab and run:

```sql
CREATE DATABASE test_database;
USE test_database;

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    email VARCHAR(100)
);

INSERT INTO users (name, email) VALUES 
('John Doe', 'john@example.com'),
('Jane Smith', 'jane@example.com');

SELECT * FROM users;
```

If this works, your installation is successful!

---

## Troubleshooting

### Common Issues

**"Access denied for user 'root'@'localhost'"**
- You're using the wrong password
- Try the password you set during installation
- If forgotten, you'll need to reset it using MySQL safe mode

**"Can't connect to MySQL server on 'localhost'"**
- MySQL service isn't running
- **Windows**: Check Services → MySQL80 is started
- **Linux**: Run `sudo systemctl start mysql`

**MySQL Workbench won't start**
- **Linux**: Missing dependencies, install with: `sudo apt install libgtkmm-3.0-1v5`
- Try running from terminal to see error messages

**Forgot root password?**
- **Windows**: Stop MySQL service, start with `--skip-grant-tables` option
- **Linux**: Use MySQL safe mode to reset password

### Service Management

**Windows (Command Prompt as Administrator)**:
```cmd
net start mysql80
net stop mysql80
```

**Linux**:
```bash
sudo systemctl start mysql
sudo systemctl stop mysql
sudo systemctl restart mysql
sudo systemctl status mysql
```

---

## ⚠️ IMPORTANT REMINDERS

1. **NEVER FORGET YOUR ROOT PASSWORD** - Write it down securely
2. **Don't use root for daily work** - Create separate users
3. **Enable firewall rules** if accessing MySQL remotely
4. **Regular backups** - Learn mysqldump command
5. **Keep MySQL updated** for security patches

---

## Quick Start Checklist

- [ ] Downloaded and installed MySQL Server
- [ ] Downloaded and installed MySQL Workbench  
- [ ] **Set and remembered root password**
- [ ] MySQL service is running
- [ ] Successfully connected via command line with `mysql -u root -p`
- [ ] Successfully connected via MySQL Workbench
- [ ] Created test database and table
- [ ] Created regular user account for daily use

**Your MySQL installation is now complete and ready for development!**