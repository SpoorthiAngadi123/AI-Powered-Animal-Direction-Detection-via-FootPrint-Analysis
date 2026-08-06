# Distribution Guide - What to Include/Exclude

This guide helps you prepare the project for distribution, reducing size from ~5.45 GB to a manageable package.

## Quick Summary

**Estimated sizes:**

- **Full project**: ~5.45 GB
- **Minimal distribution** (code + trained model only): ~50-100 MB
- **With training data** (for retraining): ~4-5 GB
- **With all outputs** (for demonstration): ~5.45 GB

---

## Option 1: Minimal Distribution (Recommended for Most Cases)

**Size: ~50-100 MB**  
**Use case:** Selling/licensing the working system, end users who just want to run predictions

### ✅ KEEP (Essential Files)

```
project/
├── api.py                          # Backend API
├── train_yolov8.py                 # Training script (optional but recommended)
├── infer.py                        # CLI inference script
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation
├── QUICKSTART.md                   # Quick start guide
├── DETAILED_DOCUMENTATION.md      # Technical documentation
├── TRAINING_README.md              # Training instructions
├── start_backend.bat               # Windows startup script
├── start_backend.sh                # Linux/Mac startup script
│
├── frontend/                       # Frontend source code
│   ├── src/
│   ├── index.html
│   ├── package.json                # ⚠️ Keep this, exclude node_modules
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── runs/classify/train/weights/
│   └── best.pt                      # ✅ TRAINED MODEL (ESSENTIAL!)
│
└── static/
    └── confusion_matrix.png         # Optional: for metrics display
```

### ❌ EXCLUDE (Can be Regenerated)

```
❌ __pycache__/                      # Python bytecode
❌ node_modules/                     # npm install regenerates this
❌ data/                             # Training data (large, ~4-5 GB)
   ❌ train/
   ❌ val/
   ❌ *.cache                        # YOLO cache files
❌ runs/classify/predict/            # Prediction outputs
❌ runs/classify/train/              # Keep only weights/best.pt
   ❌ results.csv                    # Can regenerate
   ❌ results.png                    # Can regenerate
   ❌ accuracy_loss_curves.png       # Can regenerate
❌ runs/classify/val/                # Validation outputs
❌ static/overlays/                  # Generated prediction overlays
❌ predictions.json                   # User-specific history
❌ yolov8n-cls.pt                    # Base model (downloads automatically)
```

### 📝 Instructions for Recipient

Add this to your README or create `SETUP_INSTRUCTIONS.md`:

````markdown
## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
````

2. **Install frontend dependencies:**

   ```bash
   cd frontend
   npm install
   ```

3. **Start the backend:**

   ```bash
   python api.py
   ```

4. **Start the frontend (in another terminal):**
   ```bash
   cd frontend
   npm run dev
   ```

The trained model is already included at `runs/classify/train/weights/best.pt`.

```

---

## Option 2: With Training Data (For Developers/Researchers)
**Size: ~4-5 GB**
**Use case:** Selling to developers who want to retrain or modify the model

### ✅ KEEP (Everything from Option 1, plus):

```

✅ data/
✅ train/ # Training images
✅ val/ # Validation images
✅ _.md # Dataset documentation
✅ _.py # Data preparation scripts

```

### ❌ Still EXCLUDE:

```

❌ **pycache**/
❌ node_modules/
❌ data/\*.cache # Cache files (regenerated)
❌ runs/classify/predict/ # Prediction outputs
❌ static/overlays/ # Generated overlays
❌ predictions.json # User history

```

---

## Option 3: Complete Package (For Demonstrations)
**Size: ~5.45 GB**
**Use case:** Complete archive for backup, demonstration, or academic submission

### ✅ KEEP Everything except:

```

❌ **pycache**/ # Still exclude (regenerated)
❌ node_modules/ # Still exclude (regenerated)

````

**Note:** This is the full project. Recipients will need to run `npm install` and may want to clear `static/overlays/` and `predictions.json` for a fresh start.

---

## Automated Cleanup Scripts

### Windows PowerShell Script (`cleanup_for_distribution.ps1`)

```powershell
# Remove unnecessary files/folders for distribution
Write-Host "Cleaning up project for distribution..."

# Remove Python cache
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue

# Remove node_modules (will be regenerated)
Remove-Item -Recurse -Force frontend\node_modules -ErrorAction SilentlyContinue

# Remove YOLO cache files
Remove-Item -Force data\train.cache -ErrorAction SilentlyContinue
Remove-Item -Force data\val.cache -ErrorAction SilentlyContinue

# Remove prediction outputs
Remove-Item -Recurse -Force runs\classify\predict -ErrorAction SilentlyContinue

# Remove generated overlays
Remove-Item -Recurse -Force static\overlays -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path static\overlays -Force | Out-Null

# Remove user prediction history
Remove-Item -Force predictions.json -ErrorAction SilentlyContinue

# Remove base model (can be downloaded)
Remove-Item -Force yolov8n-cls.pt -ErrorAction SilentlyContinue

# Remove training visualization outputs (optional - keep if you want to show training results)
# Remove-Item -Force runs\classify\train\results.csv -ErrorAction SilentlyContinue
# Remove-Item -Force runs\classify\train\results.png -ErrorAction SilentlyContinue
# Remove-Item -Force runs\classify\train\accuracy_loss_curves.png -ErrorAction SilentlyContinue

Write-Host "Cleanup complete!"
Write-Host "Remaining size:"
Get-ChildItem -Recurse | Measure-Object -Property Length -Sum | Select-Object @{Name="Size(GB)";Expression={[math]::Round($_.Sum / 1GB, 2)}}
````

### Linux/Mac Bash Script (`cleanup_for_distribution.sh`)

```bash
#!/bin/bash
echo "Cleaning up project for distribution..."

# Remove Python cache
rm -rf __pycache__

# Remove node_modules
rm -rf frontend/node_modules

# Remove YOLO cache files
rm -f data/train.cache data/val.cache

# Remove prediction outputs
rm -rf runs/classify/predict

# Remove generated overlays
rm -rf static/overlays
mkdir -p static/overlays

# Remove user prediction history
rm -f predictions.json

# Remove base model
rm -f yolov8n-cls.pt

echo "Cleanup complete!"
echo "Remaining size:"
du -sh .
```

---

## Creating Distribution Packages

### Option A: ZIP Archive (Recommended)

**Minimal package:**

```powershell
# Windows
Compress-Archive -Path api.py,train_yolov8.py,infer.py,requirements.txt,README.md,QUICKSTART.md,DETAILED_DOCUMENTATION.md,TRAINING_README.md,start_backend.bat,start_backend.sh,frontend,runs/classify/train/weights/best.pt,static/confusion_matrix.png -DestinationPath footprint-detection-minimal.zip
```

**With training data:**

```powershell
Compress-Archive -Path * -Exclude __pycache__,node_modules,*.cache,runs/classify/predict,static/overlays,predictions.json,yolov8n-cls.pt -DestinationPath footprint-detection-full.zip
```

### Option B: Git Repository (For Version Control)

Create a `.gitignore` if you don't have one:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Node
node_modules/
npm-debug.log*

# YOLO
*.cache
runs/classify/predict/
runs/classify/val/

# Generated files
static/overlays/
predictions.json

# Base model (download separately)
yolov8n-cls.pt

# OS
.DS_Store
Thumbs.db
```

Then:

```bash
git init
git add .
git commit -m "Initial distribution package"
```

---

## File Size Breakdown (Approximate)

| Component                             | Size           | Keep?                           |
| ------------------------------------- | -------------- | ------------------------------- |
| `data/train/`                         | ~2.5 GB        | Only if including training data |
| `data/val/`                           | ~600 MB        | Only if including training data |
| `runs/classify/predict/`              | ~500 MB        | ❌ No (regenerated)             |
| `frontend/node_modules/`              | ~200 MB        | ❌ No (npm install)             |
| `runs/classify/train/weights/best.pt` | ~6 MB          | ✅ Yes (essential)              |
| `static/overlays/`                    | ~50 MB         | ❌ No (regenerated)             |
| Source code                           | ~1 MB          | ✅ Yes                          |
| Documentation                         | ~100 KB        | ✅ Yes                          |
| **Total (minimal)**                   | **~50-100 MB** | ✅ Recommended                  |

---

## Checklist Before Distribution

- [ ] Run cleanup script to remove unnecessary files
- [ ] Verify `runs/classify/train/weights/best.pt` exists (trained model)
- [ ] Test that the project works after cleanup:
  - [ ] `pip install -r requirements.txt` works
  - [ ] `cd frontend && npm install` works
  - [ ] `python api.py` starts successfully
  - [ ] Frontend can make predictions
- [ ] Update README with setup instructions
- [ ] Create `SETUP_INSTRUCTIONS.md` if needed
- [ ] Remove any sensitive data (API keys, personal info)
- [ ] Check `predictions.json` doesn't contain sensitive information
- [ ] Create distribution archive (ZIP/tar.gz)
- [ ] Test the archive on a clean machine if possible

---

## License Considerations

If selling/licensing, consider:

- Adding a `LICENSE` file
- Adding a `NOTICE` file for third-party attributions (YOLOv8, React, etc.)
- Including terms of use in README
- Watermarking or protecting the trained model if proprietary

---

## Quick Commands Summary

**Minimal cleanup (Windows):**

```powershell
Remove-Item -Recurse -Force __pycache__,frontend\node_modules,data\*.cache,runs\classify\predict,static\overlays,predictions.json,yolov8n-cls.pt -ErrorAction SilentlyContinue
```

**Minimal cleanup (Linux/Mac):**

```bash
rm -rf __pycache__ frontend/node_modules data/*.cache runs/classify/predict static/overlays predictions.json yolov8n-cls.pt
```

**Create minimal ZIP (Windows):**

```powershell
Compress-Archive -Path api.py,train_yolov8.py,infer.py,requirements.txt,*.md,start_backend.*,frontend,runs/classify/train/weights,static/confusion_matrix.png -DestinationPath footprint-detection.zip
```

---

**Recommended:** Use **Option 1 (Minimal Distribution)** for most cases. It reduces size from 5.45 GB to ~50-100 MB while keeping everything needed to run the system.
