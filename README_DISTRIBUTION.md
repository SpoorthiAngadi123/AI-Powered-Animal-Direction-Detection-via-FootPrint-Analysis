# Footprint Direction Detection - Distribution Package

This is a clean distribution package containing only the essential files needed to run the system.

## Quick Start

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

3. Start the backend:
   ```bash
   python api.py
   ```
   The API will run on http://127.0.0.1:8001

4. Start the frontend (in another terminal):
   ```bash
   cd frontend
   npm run dev
   ```
   The frontend will run on http://localhost:5173

## What's Included

- All source code (backend and frontend)
- Trained model (runs/classify/train/weights/best.pt)
- Configuration files
- Documentation
- Startup scripts

## What's NOT Included

- Training data (not needed for inference)
- node_modules/ (run npm install to regenerate)
- Generated prediction outputs
- User prediction history

## Model Information

The trained YOLOv8n-cls model is located at:
runs/classify/train/weights/best.pt

This model classifies footprints into 4 directions:
- North
- South
- East
- West

## Troubleshooting

- Model not found error: Ensure runs/classify/train/weights/best.pt exists
- Port already in use: Change port in api.py or stop the existing process
- Frontend won't start: Run npm install in the frontend directory

## Documentation

See the included documentation files:
- README.md - Main project documentation
- QUICKSTART.md - Quick start guide
- DETAILED_DOCUMENTATION.md - Technical deep dive
- TRAINING_README.md - Training instructions

---

Package created: 2025-11-21 00:14:56
