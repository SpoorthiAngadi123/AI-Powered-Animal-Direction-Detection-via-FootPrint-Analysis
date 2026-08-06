"""
YOLOv8 Classification Training Script for Footprint Direction Classification

This script trains a YOLOv8 classification model to classify footprint directions:
- North, South, East, West

Required folder structure:
data/
  train/
    north/  (or North/)
    south/  (or South/)
    east/   (or East/)
    west/   (or West/)
  val/
    north/  (or North/)
    south/  (or South/)
    east/   (or East/)
    west/   (or West/)

Usage:
    python train_yolov8.py
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import glob
import matplotlib.pyplot as plt


def run_command(cmd, description):
    """Run a shell command and print the output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    # Split command for subprocess
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"⚠️  Warning: Command may have failed with return code {result.returncode}")
    return result


def install_dependencies():
    """Install required dependencies."""
    print("\n" + "="*60)
    print("📦 STEP 1: Installing dependencies")
    print("="*60)
    
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install ultralytics torch torchvision matplotlib seaborn --extra-index-url https://download.pytorch.org/whl/cu121", 
         "Installing YOLOv8 and PyTorch with CUDA support")
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)


def verify_environment():
    """Verify the environment and dependencies."""
    print("\n" + "="*60)
    print("✅ STEP 2: Verifying environment")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        import torch
        print("✅ Ultralytics imported successfully")
        print(f"✅ Torch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✅ Current working directory: {os.getcwd()}")
        
        # Check data structure
        data_dir = Path("data")
        if data_dir.exists():
            train_dir = data_dir / "train"
            val_dir = data_dir / "val"
            if train_dir.exists() and val_dir.exists():
                print(f"✅ Data directory structure found")
                # Check for class folders
                train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
                val_classes = [d.name for d in val_dir.iterdir() if d.is_dir()]
                print(f"✅ Train classes found: {train_classes}")
                print(f"✅ Val classes found: {val_classes}")
            else:
                print("⚠️  Warning: train/ or val/ directories not found in data/")
        else:
            print("⚠️  Warning: data/ directory not found")
            
    except ImportError as e:
        print(f"❌ Error importing dependencies: {e}")
        print("Please install dependencies first.")
        return False
    
    return True


def train_model():
    """Train the YOLOv8 classification model."""
    print("\n" + "="*60)
    print("🚀 STEP 3: Training YOLOv8 classification model")
    print("="*60)
    print("Note: augment=False keeps directional labels correct (no flips/rotations)")
    
    cmd = (
        "yolo classify train "
        "data=data "
        "model=yolov8n-cls.pt "
        "epochs=15 "
        "imgsz=224 "
        "batch=32 "
        "lr0=0.001 "
        "dropout=0.2 "
        "patience=5 "
        "augment=False "
        "seed=42"
    )
    
    run_command(cmd, "Training YOLOv8 classification model")


def validate_model():
    """Validate the trained model and generate metrics."""
    print("\n" + "="*60)
    print("📊 STEP 4: Validating model and generating metrics")
    print("="*60)
    
    model_path = "runs/classify/train/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first.")
        return False
    
    cmd = f"yolo classify val model={model_path} data=data plots=True"
    run_command(cmd, "Validating model and generating confusion matrix")
    return True


def predict_validation():
    """Run predictions on validation set."""
    print("\n" + "="*60)
    print("🔮 STEP 5: Predicting on validation set")
    print("="*60)
    
    model_path = "runs/classify/train/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        return False
    
    # YOLOv8 predict doesn't automatically recurse into subdirectories
    # So we predict on each class folder separately
    val_dir = Path("data/val")
    if not val_dir.exists():
        print(f"❌ Error: Validation directory not found at {val_dir}")
        return False
    
    class_folders = [d for d in val_dir.iterdir() if d.is_dir()]
    if not class_folders:
        print(f"❌ Error: No class folders found in {val_dir}")
        return False
    
    print(f"Found {len(class_folders)} class folders: {[f.name for f in class_folders]}")
    
    # Use Python API for more reliable subdirectory handling
    try:
        from ultralytics import YOLO
        
        print("Loading model...")
        model = YOLO(model_path)
        
        total_images = 0
        for class_folder in class_folders:
            # Count images in this folder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp']
            images = []
            for ext in image_extensions:
                images.extend(class_folder.glob(ext))
                images.extend(class_folder.glob(ext.upper()))
            
            if images:
                print(f"Predicting on {class_folder.name} ({len(images)} images)...")
                results = model.predict(
                    source=str(class_folder),
                    imgsz=224,
                    save=True,
                    project="runs/classify/predict",
                    name=f"val_{class_folder.name}"
                )
                total_images += len(results)
            else:
                print(f"⚠️  Warning: No images found in {class_folder.name}")
        
        print(f"\n✅ Predictions completed on {total_images} images total")
        print("✅ Predictions saved inside: runs/classify/predict/")
        return True
        
    except ImportError:
        print("⚠️  Ultralytics not available, trying command-line approach...")
        # Fallback to command-line
        for class_folder in class_folders:
            cmd = (
                f"yolo classify predict "
                f"model={model_path} "
                f"source={class_folder} "
                f"imgsz=224 "
                f"save=True "
                f"project=runs/classify/predict "
                f"name=val_{class_folder.name}"
            )
            run_command(cmd, f"Predicting on {class_folder.name}")
        
        print("✅ Predictions saved inside: runs/classify/predict/")
        return True
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_metrics():
    """Generate visualization plots for training metrics."""
    print("\n" + "="*60)
    print("📈 STEP 6: Generating visualization plots")
    print("="*60)
    
    try:
        import pandas as pd
        
        # Try CSV first (YOLOv8 standard format)
        csv_file = "runs/classify/train/results.csv"
        json_file = "runs/classify/train/results.json"
        
        if Path(csv_file).exists():
            print(f"Reading metrics from: {csv_file}")
            df = pd.read_csv(csv_file)
            
            # Get column names
            epoch_col = None
            train_loss_col = None
            train_acc_col = None
            val_acc_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'epoch' in col_lower:
                    epoch_col = col
                elif 'loss' in col_lower and 'train' in col_lower:
                    train_loss_col = col
                elif 'accuracy' in col_lower or 'acc' in col_lower:
                    if 'train' in col_lower or 'top1' in col_lower:
                        if train_acc_col is None:
                            train_acc_col = col
                    elif 'val' in col_lower:
                        val_acc_col = col
            
            # Fallback: try common column names
            if epoch_col is None:
                epoch_col = df.columns[0] if len(df.columns) > 0 else None
            if train_loss_col is None:
                for col in df.columns:
                    if 'loss' in col.lower():
                        train_loss_col = col
                        break
            if train_acc_col is None:
                for col in df.columns:
                    if ('accuracy' in col.lower() or 'acc' in col.lower()) and 'val' not in col.lower():
                        train_acc_col = col
                        break
            if val_acc_col is None:
                for col in df.columns:
                    if 'val' in col.lower() and ('accuracy' in col.lower() or 'acc' in col.lower()):
                        val_acc_col = col
                        break
            
            if epoch_col is None:
                print("⚠️  Warning: Could not find epoch column in CSV")
                return False
            
            epochs = df[epoch_col].values if epoch_col else range(len(df))
            
            plt.figure(figsize=(12, 4))
            
            # Plot loss if available
            if train_loss_col:
                plt.subplot(1, 2, 1)
                plt.plot(epochs, df[train_loss_col].values, label="Train Loss", linewidth=2)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training Loss Curve")
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Plot accuracy if available
            if train_acc_col or val_acc_col:
                plt.subplot(1, 2, 2 if train_loss_col else 1)
                if train_acc_col:
                    plt.plot(epochs, df[train_acc_col].values, label="Train Accuracy", linewidth=2)
                if val_acc_col:
                    plt.plot(epochs, df[val_acc_col].values, label="Val Accuracy", linewidth=2)
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title("Accuracy Curve")
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = "runs/classify/train/accuracy_loss_curves.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Accuracy & loss curves saved to {output_path}")
            plt.close()
            return True
            
        elif Path(json_file).exists():
            print(f"Reading metrics from: {json_file}")
            with open(json_file, "r") as f:
                metrics = json.load(f)
            
            epochs = [x["epoch"] for x in metrics]
            train_acc = [x.get("metrics/accuracy_top1", 0) for x in metrics]
            val_acc = [x.get("metrics/val/accuracy_top1", 0) for x in metrics]
            train_loss = [x.get("train/loss", 0) for x in metrics]
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
            plt.plot(epochs, val_acc, label="Val Accuracy", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = "runs/classify/train/accuracy_loss_curves.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Accuracy & loss curves saved to {output_path}")
            plt.close()
            return True
        else:
            print("⚠️  Warning: No results.csv or results.json file found.")
            print("   YOLOv8 may have already generated results.png - check runs/classify/train/results.png")
            return False
        
    except ImportError:
        print("⚠️  Warning: pandas not available. Install with: pip install pandas")
        print("   YOLOv8 may have already generated results.png - check runs/classify/train/results.png")
        return False
    except Exception as e:
        print(f"⚠️  Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_inference_script():
    """Create the inference script for making predictions."""
    print("\n" + "="*60)
    print("📝 STEP 7: Creating inference script")
    print("="*60)
    
    infer_code = """from ultralytics import YOLO
import sys
from pathlib import Path

# Class names - adjust if your folders use different capitalization
CLASSES = ['north', 'south', 'east', 'west']

def main(img_path):
    model_path = Path('runs/classify/train/weights/best.pt')
    
    if not model_path.exists():
        print(f'Error: Model not found at {model_path}')
        print('Please train the model first using train_yolov8.py')
        sys.exit(1)
    
    model = YOLO(str(model_path))
    results = model(img_path, imgsz=224)
    probs = results[0].probs
    
    pred_idx = int(probs.top1)
    conf = float(probs.top1conf)
    
    # Get class names from model if available
    if hasattr(results[0], 'names'):
        class_names = results[0].names
        pred_class = class_names[pred_idx]
    else:
        pred_class = CLASSES[pred_idx] if pred_idx < len(CLASSES) else f'class_{pred_idx}'
    
    print(f'Predicted direction: {pred_class} (confidence={conf:.3f})')
    
    # Print all class probabilities
    if hasattr(probs, 'data'):
        print('\\nAll class probabilities:')
        for i, prob in enumerate(probs.data):
            class_name = class_names[i] if hasattr(results[0], 'names') else (CLASSES[i] if i < len(CLASSES) else f'class_{i}')
            print(f'  {class_name}: {prob:.3f}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python infer.py <path_to_image>')
        sys.exit(1)
    
    img_path = sys.argv[1]
    if not Path(img_path).exists():
        print(f'Error: Image not found at {img_path}')
        sys.exit(1)
    
    main(img_path)
"""
    
    with open("infer.py", "w") as f:
        f.write(infer_code)
    
    print("✅ infer.py created!")
    print("   Run: python infer.py path/to/test_image.jpg")


def export_onnx():
    """Export the trained model to ONNX format."""
    print("\n" + "="*60)
    print("📦 STEP 8: Exporting model to ONNX format")
    print("="*60)
    
    model_path = "runs/classify/train/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        return False
    
    cmd = f"yolo export model={model_path} format=onnx"
    run_command(cmd, "Exporting model to ONNX format")
    
    onnx_path = "runs/classify/train/weights/best.onnx"
    if Path(onnx_path).exists():
        print(f"✅ Exported model to ONNX format: {onnx_path}")
    else:
        print("⚠️  Warning: ONNX file not found after export")
    
    return True


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("🚀 YOLOv8 Footprint Direction Classification Training")
    print("="*60)
    
    # Step 1: Install dependencies (optional - comment out if already installed)
    install_choice = input("\nInstall dependencies? (y/n, default=n): ").strip().lower()
    if install_choice == 'y':
        install_dependencies()
    else:
        print("⏭️  Skipping dependency installation")
    
    # Step 2: Verify environment
    if not verify_environment():
        print("❌ Environment verification failed. Please check your setup.")
        return
    
    # Step 3: Train model
    train_choice = input("\nTrain the model? (y/n, default=y): ").strip().lower()
    if train_choice != 'n':
        train_model()
    else:
        print("⏭️  Skipping training")
    
    # Step 4: Validate model
    validate_choice = input("\nValidate the model? (y/n, default=y): ").strip().lower()
    if validate_choice != 'n':
        validate_model()
    
    # Step 5: Predict on validation set
    predict_choice = input("\nRun predictions on validation set? (y/n, default=y): ").strip().lower()
    if predict_choice != 'n':
        predict_validation()
    
    # Step 6: Visualize metrics
    visualize_choice = input("\nGenerate visualization plots? (y/n, default=y): ").strip().lower()
    if visualize_choice != 'n':
        visualize_metrics()
    
    # Step 7: Create inference script
    create_inference_script()
    
    # Step 8: Export to ONNX
    export_choice = input("\nExport model to ONNX? (y/n, default=y): ").strip().lower()
    if export_choice != 'n':
        export_onnx()
    
    print("\n" + "="*60)
    print("🎯 DONE: Training pipeline completed!")
    print("="*60)
    print("\n📁 Output files:")
    print("   - Model weights: runs/classify/train/weights/best.pt")
    print("   - Validation results: runs/classify/val/")
    print("   - Predictions: runs/classify/predict/")
    print("   - Inference script: infer.py")
    print("\n💡 Next steps:")
    print("   - Test inference: python infer.py path/to/test_image.jpg")
    print("   - Check confusion matrix: runs/classify/val/confusion_matrix.png")
    print("   - View training curves: runs/classify/train/accuracy_loss_curves.png")


if __name__ == "__main__":
    main()

