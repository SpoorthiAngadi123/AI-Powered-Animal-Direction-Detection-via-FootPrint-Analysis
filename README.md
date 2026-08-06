# Footprint Direction Classification

A full-stack application for classifying animal footprint directions (North, South, East, West) using YOLOv8 classification model.

## Project Structure

```
.
├── api.py                 # FastAPI backend server
├── train_yolov8.py        # Training script
├── infer.py               # CLI inference script
├── requirements.txt       # Python dependencies
├── frontend/              # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx      # Image upload and prediction
│   │   │   └── Metrics.jsx   # Model metrics and prediction history
│   └── package.json
├── data/                  # Training data
│   ├── train/
│   │   ├── North/
│   │   ├── South/
│   │   ├── East/
│   │   └── West/
│   └── val/
│       ├── North/
│       ├── South/
│       ├── East/
│       └── West/
└── runs/                  # Training outputs
    └── classify/
        └── train/
            └── weights/
                └── best.pt  # Trained model
```

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Train the Model (if not already trained)

```bash
python train_yolov8.py
```

## Running the Application

### Option 1: Run Backend and Frontend Separately

**Terminal 1 - Start Backend API:**
```bash
python api.py
```
The API will run on `http://127.0.0.1:8001`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```
The frontend will run on `http://localhost:5173` (or another port if 5173 is taken)

### Option 2: Use the Startup Script

**Windows:**
```bash
start_backend.bat
```

**Linux/Mac:**
```bash
chmod +x start_backend.sh
./start_backend.sh
```

Then in another terminal:
```bash
cd frontend
npm run dev
```

## Usage

1. **Upload an Image**: Go to the Home page and upload a footprint image
2. **View Prediction**: The model will predict the direction (North, South, East, West) with confidence score
3. **View Metrics**: Go to the Metrics page to see:
   - Overall model accuracy
   - Confusion matrix
   - Recent prediction history

## API Endpoints

- `POST /predict` - Upload an image and get prediction
- `GET /metrics` - Get model metrics from validation results
- `GET /predictions` - Get prediction history (last 100 predictions)
- `GET /static/*` - Serve static files (overlay images, confusion matrix)

## Model Information

- **Model**: YOLOv8n-cls (nano classification model)
- **Classes**: North, South, East, West
- **Input Size**: 224x224
- **Training**: See `TRAINING_README.md` for details

## Troubleshooting

### Backend Issues

- **Model not found**: Make sure you've trained the model first using `train_yolov8.py`
- **Port already in use**: Change the port in `api.py` (line with `uvicorn.run`)

### Frontend Issues

- **CORS errors**: Make sure the backend is running and the CORS origins in `api.py` match your frontend URL
- **API connection failed**: Check that `VITE_API_BASE` environment variable is set correctly, or the default `http://127.0.0.1:8001` is correct

### Common Issues

- **Image upload fails**: Make sure the image is a valid format (jpg, png, webp, etc.)
- **Metrics not loading**: Ensure the validation results exist in `runs/classify/val/`

## Development

### Backend Development

The backend uses FastAPI. To see the API documentation:
- Swagger UI: `http://127.0.0.1:8001/docs`
- ReDoc: `http://127.0.0.1:8001/redoc`

### Frontend Development

The frontend uses React with Vite. Hot reload is enabled during development.

## Production Deployment

For production:
1. Build the frontend: `cd frontend && npm run build`
2. Serve the built files with a web server (nginx, etc.)
3. Run the backend with a production ASGI server (gunicorn with uvicorn workers)
4. Set up proper CORS origins in `api.py`

