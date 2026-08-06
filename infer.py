from ultralytics import YOLO
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
        print('\nAll class probabilities:')
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
