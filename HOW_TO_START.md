# X-Fusion - How to Start the Project

X-Fusion is a multimodal AI system for automated pneumonia detection from chest X-rays.
It uses BiomedCLIP + CXR-BERT fusion with a FastAPI backend and a Next.js frontend.

---

## Prerequisites

| Tool    | Version | Notes               |
|---------|---------|---------------------|
| Python  | >= 3.10 | For the backend     |
| Node.js | >= 18.x | For the frontend    |
| npm     | >= 9.x  | Comes with Node.js  |

---

## Project Structure

  3 Project/
  +-- backend/              # FastAPI inference server
  |   +-- main.py           # API entry point (uvicorn backend.main:app)
  |   +-- models/           # MultimodalModel (BiomedCLIP + CXR-BERT)
  |   +-- pipelines/        # Training & evaluation pipelines
  |   +-- scripts/          # Dataset setup scripts
  |   +-- checkpoints/      # Trained model weights (.ckpt)
  |   \-- train.py          # Training script
  +-- frontend/             # Next.js 16 web UI
  |   \-- src/app/          # App router pages & components
  +-- documents/            # Papers and project documents
  |   +-- latex/            # LaTeX sources
  |   +-- pdf/              # PDF exports
  |   \-- word/             # Word documents and templates
  +-- requirements.txt      # Python dependencies
  \-- HOW_TO_START.md       # This file

---

## First-Time Setup

### 1. Open a terminal in the root project folder (3 Project/)

### 2. Create and activate Python virtual environment

  # Create venv (only once)
  python -m venv venv

  # Activate (PowerShell)
  venv\Scripts\Activate.ps1

  # Activate (CMD)
  venv\Scripts\activate.bat

### 3. Install Python dependencies

  pip install -r requirements.txt

### 4. Install frontend dependencies

  cd frontend
  npm install
  cd ..

---

## Running the Project

You need TWO terminals - one for backend, one for frontend.

---

### Terminal 1 - Backend (FastAPI)

Open a terminal in the ROOT project folder (3 Project/):

  # Activate virtual environment first
  venv\Scripts\Activate.ps1

  # Start the backend server
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Backend is ready when you see:
  INFO:  Uvicorn running on http://0.0.0.0:8000

Available endpoints:
  GET  http://localhost:8000/health    - Health check
  POST http://localhost:8000/predict   - Run inference on a chest X-ray
  GET  http://localhost:8000/docs      - Swagger UI (auto-generated)

---

### Terminal 2 - Frontend (Next.js)

Open a NEW terminal in the frontend/ folder:

  cd frontend
  npm run dev

Frontend is ready when you see:
  Local:   http://localhost:3000
  Ready in ~5s

Open your browser at: http://localhost:3000

---

## (Optional) Training the Model

  # Activate venv first
  venv\Scripts\Activate.ps1

  # Download and prepare dataset (requires Kaggle API key)
  python backend/scripts/setup_dataset.py

  # Train the model
  python -m backend.train

  # For advanced/cluster training
  python -m backend.train_advanced_laptop
  python -m backend.train_cluster

Trained checkpoints are saved to backend/checkpoints/
The backend automatically loads the latest best checkpoint on startup.

---

## Environment Variables (Optional)

  KAGGLE_USERNAME    Kaggle username for dataset download
  KAGGLE_KEY         Kaggle API key for dataset download

---

## Stopping the Servers

In each terminal press: Ctrl + C

---

## Troubleshooting

| Problem                             | Fix                                                                      |
|-------------------------------------|--------------------------------------------------------------------------|
| ModuleNotFoundError: backend        | Run uvicorn from the ROOT folder (3 Project/), not from backend/         |
| Backend fails during startup        | Add a compatible `.ckpt` file under `backend/checkpoints/`, then restart   |
| Port 8000 already in use            | Use --port 8001 and update frontend API URL                              |
| Port 3000 already in use            | Next.js auto-picks 3001, 3002...                                         |
| Activate.ps1 blocked                | Run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned                 |

---

## API Quick Test (PowerShell)

  # Health check
  Invoke-RestMethod http://localhost:8000/health

  # Or with curl
  curl http://localhost:8000/health

---

Generated for X-Fusion - DAViT Multimodal Pneumonia Detection System
