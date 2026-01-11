# 🚀 COMPLETE GITHUB REPOSITORY SETUP GUIDE
## Sensitivity-Theoretic Compiler Testing Framework

---

## 📋 REPOSITORY INFORMATION

| Item | Value |
|------|-------|
| **Repository Name** | `sensitivity-compiler-testing` |
| **Full URL** | `https://github.com/hmshujaatzaheer/sensitivity-compiler-testing` |
| **Description** | A novel framework for compiler testing using chaos theory and dynamical systems analysis to identify bug-prone regions. Implements Lyapunov exponent computation, phase transition detection, and PAC learning-based test prioritization. |
| **Topics/Tags** | `compiler-testing`, `chaos-theory`, `lyapunov-exponents`, `fuzzing`, `software-testing`, `dynamical-systems`, `pac-learning`, `gcc`, `llvm`, `bug-detection` |

---

## 📁 STEP 1: DOWNLOAD THE REPOSITORY FILES

First, download all the repository files I created. They are available in the outputs.

---

## 💻 STEP 2: INSTALL GIT (If Not Already Installed)

### Windows PowerShell Commands:

```powershell
# Check if Git is installed
git --version

# If not installed, install via winget (Windows 10/11)
winget install --id Git.Git -e --source winget

# OR download from: https://git-scm.com/download/win
```

---

## 🔧 STEP 3: CONFIGURE GIT (First Time Only)

Open **PowerShell as Administrator** and run:

```powershell
# Set your name (use your full name)
git config --global user.name "H M Shujaat Zaheer"

# Set your email (MUST match GitHub email)
git config --global user.email "shujabis@gmail.com"

# Verify configuration
git config --global --list

# Set default branch name to 'main'
git config --global init.defaultBranch main

# Enable credential helper (remembers your password)
git config --global credential.helper manager
```

---

## 🌐 STEP 4: CREATE REPOSITORY ON GITHUB (Website)

### 4.1 Go to GitHub
Open your web browser and go to: `https://github.com/hmshujaatzaheer`

### 4.2 Click "New Repository" (Green Button)

### 4.3 Fill in the Details:

| Field | Value |
|-------|-------|
| Repository name | `sensitivity-compiler-testing` |
| Description | A novel framework for compiler testing using chaos theory and dynamical systems analysis to identify bug-prone regions. Implements Lyapunov exponent computation, phase transition detection, and PAC learning-based test prioritization. |
| Visibility | ✅ Public |
| Initialize with README | ❌ NO (leave unchecked - we have our own) |
| Add .gitignore | ❌ NO (leave unchecked - we have our own) |
| Add license | ❌ NO (leave unchecked - we have our own) |

### 4.4 Click "Create Repository"

---

## 📂 STEP 5: PREPARE LOCAL FOLDER

### 5.1 Create Project Folder

```powershell
# Navigate to where you want to create the project
cd C:\Users\$env:USERNAME\Documents

# Create the project folder
New-Item -ItemType Directory -Name "sensitivity-compiler-testing" -Force

# Navigate into the folder
cd sensitivity-compiler-testing
```

### 5.2 Copy All Downloaded Files Here

Copy all the files from the downloaded repository to this folder. The structure should look like:

```
sensitivity-compiler-testing/
├── src/
│   └── sensitivity_testing/
│       ├── __init__.py
│       ├── framework.py
│       ├── cli.py
│       ├── algorithms/
│       ├── core/
│       ├── oracles/
│       ├── analysis/
│       └── utils/
├── tests/
├── examples/
├── experiments/
├── docs/
├── scripts/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
├── setup.py
├── pyproject.toml
└── .gitignore
```

---

## 🔄 STEP 6: INITIALIZE GIT AND COMMIT

Run these commands **one by one** in PowerShell:

```powershell
# Step 6.1: Initialize Git repository
git init

# Step 6.2: Check status (should show all files as untracked)
git status

# Step 6.3: Add ALL files to staging
git add .

# Step 6.4: Check status again (should show files ready to commit)
git status

# Step 6.5: Create your first commit
git commit -m "Initial commit: Sensitivity-theoretic compiler testing framework

- Implemented DiscreteLyapunov algorithm (O(T log T)) for sensitivity analysis
- Implemented PhaseTransitionDetector with CUSUM, PELT, and BOCPD methods
- Implemented SensitivityOracle with PAC learning bounds
- Added comprehensive test suite
- Added CLI interface
- Added documentation and examples"
```

---

## 🔗 STEP 7: CONNECT TO GITHUB AND PUSH

```powershell
# Step 7.1: Add the remote GitHub repository
git remote add origin https://github.com/hmshujaatzaheer/sensitivity-compiler-testing.git

# Step 7.2: Verify remote is added
git remote -v

# Step 7.3: Push to GitHub (first time with -u flag)
git push -u origin main
```

### If Asked for Authentication:
- **Username**: `hmshujaatzaheer`
- **Password**: Use a Personal Access Token (PAT), NOT your GitHub password

### How to Create Personal Access Token:
1. Go to: `https://github.com/settings/tokens`
2. Click "Generate new token (classic)"
3. Name: "Repository Access"
4. Expiration: 90 days
5. Select scopes: ✅ `repo` (all repo permissions)
6. Click "Generate token"
7. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
8. Use this token as your password when pushing

---

## 🏷️ STEP 8: ADD TOPICS/TAGS ON GITHUB

### 8.1 Go to your repository page:
`https://github.com/hmshujaatzaheer/sensitivity-compiler-testing`

### 8.2 Click the ⚙️ gear icon next to "About"

### 8.3 Add these topics (one by one):
```
compiler-testing
chaos-theory
lyapunov-exponents
fuzzing
software-testing
dynamical-systems
pac-learning
gcc
llvm
bug-detection
python
research
```

### 8.4 Click "Save changes"

---

## ✅ STEP 9: VERIFY EVERYTHING

```powershell
# Check repository status
git status

# Check commit history
git log --oneline

# Check remote connection
git remote -v
```

Visit: `https://github.com/hmshujaatzaheer/sensitivity-compiler-testing`

You should see all your files!

---

## 🔄 STEP 10: FUTURE UPDATES

Whenever you make changes:

```powershell
# Check what changed
git status

# Add changed files
git add .

# Commit with message
git commit -m "Your descriptive message here"

# Push to GitHub
git push
```

---

## 📝 COMPLETE POWERSHELL SCRIPT (All-in-One)

Save this as `setup_repo.ps1` and run it:

```powershell
# ============================================
# COMPLETE REPOSITORY SETUP SCRIPT
# Run this in PowerShell after copying files
# ============================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Git Repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Configure Git (if not already done)
Write-Host "`n[1/6] Configuring Git..." -ForegroundColor Yellow
git config --global user.name "H M Shujaat Zaheer"
git config --global user.email "shujabis@gmail.com"
git config --global init.defaultBranch main

# Initialize repository
Write-Host "`n[2/6] Initializing repository..." -ForegroundColor Yellow
git init

# Add all files
Write-Host "`n[3/6] Adding files..." -ForegroundColor Yellow
git add .

# Show status
Write-Host "`n[4/6] Current status:" -ForegroundColor Yellow
git status

# Commit
Write-Host "`n[5/6] Creating commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Sensitivity-theoretic compiler testing framework

Features:
- DiscreteLyapunov algorithm (O(T log T)) for sensitivity analysis
- PhaseTransitionDetector with CUSUM, PELT, and BOCPD methods  
- SensitivityOracle with PAC learning bounds
- Comprehensive test suite and CLI interface
- Full documentation and examples"

# Add remote and push
Write-Host "`n[6/6] Connecting to GitHub..." -ForegroundColor Yellow
git remote add origin https://github.com/hmshujaatzaheer/sensitivity-compiler-testing.git

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "LOCAL SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nNow run: git push -u origin main" -ForegroundColor Cyan
Write-Host "You will be prompted for your GitHub credentials." -ForegroundColor Cyan
Write-Host "`nUse your Personal Access Token as the password!" -ForegroundColor Yellow
```

---

## 🎯 QUICK REFERENCE COMMANDS

| Task | Command |
|------|---------|
| Check status | `git status` |
| Add all changes | `git add .` |
| Commit changes | `git commit -m "message"` |
| Push to GitHub | `git push` |
| Pull from GitHub | `git pull` |
| View history | `git log --oneline` |
| Create branch | `git checkout -b branch-name` |
| Switch branch | `git checkout branch-name` |
| Merge branch | `git merge branch-name` |

---

## 🆘 TROUBLESHOOTING

### Error: "remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/hmshujaatzaheer/sensitivity-compiler-testing.git
```

### Error: "failed to push some refs"
```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Error: "Permission denied"
- Make sure you're using Personal Access Token, not password
- Regenerate token if expired

---

## 📧 CONTACT

- **Author**: H. M. Shujaat Zaheer
- **Email**: shujabis@gmail.com
- **GitHub**: https://github.com/hmshujaatzaheer

---

**Congratulations! Your repository is now live on GitHub! 🎉**
