# Quick Start Guide

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Trained model at `runs/classify/train/weights/best.pt`

## Step 1: Install Dependencies

### Backend
```bash
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## Step 2: Start the Backend

```bash
python api.py
```

The API will start on `http://127.0.0.1:8001`

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8001
```

## Step 3: Start the Frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:5173` (or another port)

## Step 4: Use the Application

1. Open your browser and go to `http://localhost:5173`
2. Upload an image on the Home page
3. View the prediction results
4. Check the Metrics page for model performance

## Troubleshooting

### Backend won't start
- Make sure the model exists: `runs/classify/train/weights/best.pt`
- Check if port 8001 is already in use
- Install all dependencies: `pip install -r requirements.txt`

### Frontend can't connect to backend
- Make sure the backend is running
- Check the browser console for CORS errors
- Verify the API URL in the frontend (default: `http://127.0.0.1:8001`)

### No predictions showing
- Make sure you've uploaded an image
- Check the browser console for errors
- Verify the backend logs for errors

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://127.0.0.1:8001/docs
- ReDoc: http://127.0.0.1:8001/redoc

