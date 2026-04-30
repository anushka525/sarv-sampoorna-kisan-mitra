# Frontend

This folder contains the React + Vite UI for the crop disease detector.

## Install

From inside `frontend`:

```powershell
npm install
```

## Run

```powershell
npm run dev
```

The app runs on `http://127.0.0.1:5173` and proxies `/api/*` requests to the Flask backend on `http://127.0.0.1:5000`.

## Backend expectation

The Flask backend should already be running in the project root:

```powershell
python app.py
```

The frontend calls:
- `GET /api/health`
- `POST /api/predict?top_k=3`
