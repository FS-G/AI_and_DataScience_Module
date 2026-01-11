**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*


---

# Python and VS Code Installation Guide

This guide will walk you through the installation of Python and Visual Studio Code (VS Code) on a Windows PC.

## Step 1: Download and Install Python

### 1.1 Download Python
1. Open your browser and go to the official Python website: [Python Releases for Windows](https://www.python.org/downloads/windows/)
2. Click on the “Download Python 3.x.x” button (the latest version will appear automatically).

### 1.2 Install Python
1. Run the downloaded `.exe` file (e.g., `python-3.12.0.exe`).
2. On the setup screen:
   - Check the box: "Add Python 3.x to PATH"
   - Click “Install Now”
3. Wait for the installation to complete.
4. Once installed, click “Close”

### 1.3 Verify Python Installation
1. Open Command Prompt (cmd):
   - Press `Win + R`, type `cmd`, and press Enter.
2. Type the following command and press Enter:
   ```
   python --version
   ```
   You should see something like:
   ```
   Python 3.12.0
   ```

## Step 2: Download and Install Visual Studio Code (VS Code)

### 2.1 Download VS Code
1. Visit the official VS Code website: [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Click on the “Download for Windows” button.

### 2.2 Install VS Code
1. Run the downloaded `.exe` file (e.g., `VSCodeUserSetup-x64-1.x.x.exe`).
2. Follow the setup wizard:
   - Accept the agreement and click Next
   - Choose the destination folder (default is fine) and click Next
   - Check the following boxes:
     - "Add to PATH (available after restart)"
     - "Register Code as an editor for supported file types"
     - "Add 'Open with Code' action to Windows Explorer"
   - Click Next, then Install
3. Click Finish to launch VS Code.

## Step 3: Install Python Extension in VS Code
1. Open Visual Studio Code
2. Click on the Extensions icon on the left sidebar (or press `Ctrl + Shift + X`)
3. In the search bar, type:
   ```
   Python
   ```
4. Install the official Python extension by Microsoft.

## Step 4: Test Your Python Setup in VS Code
1. Open VS Code
2. Click on File > New File
3. Save the file as `test.py`
4. Type the following code:
   ```python
   print("Hello, Python!")
   ```
5. Right-click inside the editor and select “Run Python File in Terminal”
   You should see the output:
   ```
   Hello, Python!
   ```

## Tips
- Use `Ctrl + ~` to open the terminal in VS Code.
- Install other useful extensions like Pylance, Jupyter, and Code Runner for an enhanced experience.
- Always ensure that Python is added to your PATH during installation.

---

**Course Created by: Farhan Siddiqui**  
*Data Science & AI Development Expert*