# AI-Powered Smart Trash Bin for Infant Detection

This project uses AI and image processing to detect if an infant is near or inside a trash bin to prevent accidental disposal. It ensures safety and promotes smarter waste management.

## Features
- Detects infants using a trained CNN model
- Uses image input (camera or files)
- Prevents bin operation if a baby is detected

## Technologies
- Python
- TensorFlow
- OpenCV

## Usage
1. Add your trained model in `model/trained_model.h5`
2. Add test image in `dataset/example.jpg`
3. Run:
    python src/detect_infant.py
