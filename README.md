# Sign Language Web

A FastAPI backend and static frontend web app for sign language recognition.
The system supports both **BSL** and **WLASL2000** deployment modes, with ensemble inference and video upload support.

## 📁 Repository structure

- `backend/`
  - `app/` - FastAPI application, API routes and inference logic
  - `app/checkpoints/` - model checkpoint files for BSL and WLASL2000
  - `app/class_lists/` - label list files
  - `requirements.txt` - Python dependencies
  - `run_dev.bat`, `run_dev.ps1` - development startup scripts
  - `sign/` - local Python virtual environment (do not commit)
- `frontend/`
  - `index.html` - frontend entry page
  - `assets/` - built static assets used by the UI
  - `SETUP_BACKEND_CONFIG.md` - frontend/backend config notes
- `.gitignore` - repository ignore rules

## 🚀 Quick start

### 1. Backend setup

```powershell
cd backend
python -m venv sign
.\sign\Scripts\activate
pip install -r requirements.txt
```

### 2. Set deployment mode

```powershell
$env:SL_DEPLOYMENT_MODE = "bsl"
# hoặc
$env:SL_DEPLOYMENT_MODE = "wlasl2000"
```

The backend uses this environment variable to select dataset mode and class dimensions.

### 3. Run the backend

```powershell
python -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 4. Run the frontend

```powershell
cd ..\frontend
python -m http.server 3000
```

Open the UI at:

```text
http://localhost:3000
```

### 5. Run both servers together

From the `backend` folder, use:

```cmd
run_dev.bat
```

This opens one terminal for the backend and one terminal for the frontend.

## ✅ Notes

- The backend entry point is `app.main:create_app`.
- The frontend folder is currently a static site served from `frontend/index.html` and `frontend/assets`.
- `SL_DEPLOYMENT_MODE` must match the dataset used by the frontend/UI.
- WLASL2000 strict mode requires all ensemble checkpoints in `backend/app/checkpoints/wlasl2000/`.

## ⚠️ Files that should not be committed to GitHub

- `backend/sign/` - local Python virtual environment
- `backend/sign_big.rar` - large archive file over 1.6GB
- `frontend/node_modules/` - if frontend dependencies are installed later
- Generated Python cache and logs: `__pycache__/`, `*.pyc`, `*.log`
- OS/editor files: `.DS_Store`, `Thumbs.db`, `.vscode/`, `.idea/`

## 📌 Recommended GitHub workflow

- Commit source code, backend app logic, static frontend assets, and config docs.
- Keep local environments and generated files out of version control.
- If model checkpoints are too large, consider an external release or Git LFS.

## 📚 Useful files

- `backend/requirements.txt` - Python dependencies for the backend
- `backend/run_dev.bat`, `backend/run_dev.ps1` - startup scripts for development
- `frontend/SETUP_BACKEND_CONFIG.md` - frontend/backend configuration tips
