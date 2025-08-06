# YOLO Weight Display Reader

A sophisticated computer vision system that uses YOLO object detection and OCR to automatically detect and read weight values from digital display boards, scales, and monitors.

## Features

- **Multi-Method Detection**: Uses multiple approaches to ensure accurate weight detection
  - YOLO object detection for display identification
  - Whole image OCR analysis
  - Grid-based region scanning
- **Advanced OCR**: EasyOCR integration with preprocessing for better text recognition
- **Multiple Weight Units**: Supports kg, lbs, grams, and pounds
- **Real-time Processing**: Can process both images and videos
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Visualization**: Draws bounding boxes and detected weights on images

## Installation

### Method 1: Automated Setup (Recommended)
```bash
cd yolo
python setup.py
```

### Method 2: Manual Installation
```bash
pip install -r requirements.txt
```

### Method 3: Individual Packages
```bash
pip install ultralytics opencv-python torch easyocr numpy
```

## Dependencies

- **ultralytics**: YOLOv8 implementation
- **opencv-python**: Computer vision operations
- **torch**: PyTorch for deep learning
- **easyocr**: Optical Character Recognition
- **numpy**: Numerical computations

## Usage

### Command Line Interface

#### Process an Image
```bash
python yolo.py --input weight_display.jpg --output result.jpg
```

#### Process a Video
```bash
python yolo.py --input weight_video.mp4 --output result.mp4 --video
```

#### Use GPU Acceleration
```bash
python yolo.py --input image.jpg --gpu
```

#### Custom YOLO Model
```bash
python yolo.py --input image.jpg --model yolov8m.pt --confidence 0.7
```

### Python API

```python
from yolo import WeightDisplayReader

# Initialize the reader
reader = WeightDisplayReader(
    yolo_model_path="yolov8n.pt",
    confidence_threshold=0.5,
    use_gpu=True
)

# Process an image
results = reader.process_image("weight_display.jpg")

# Print results
for result in results:
    print(f"Weight: {result['value']:.2f} {result['unit']}")
    print(f"Confidence: {result['ocr_confidence']:.2f}")
    print(f"Method: {result['detection_method']}")

# Visualize results
reader.visualize_results("weight_display.jpg", results, "output.jpg")
```

## Detection Methods

The system uses multiple detection methods to ensure accuracy:

### 1. YOLO Display Detection
- Detects screens, monitors, displays using YOLOv8
- Focuses OCR on detected display regions
- Most accurate for clear display devices

### 2. Whole Image Analysis
- Applies OCR to the entire image
- Fallback when no displays are detected
- Good for simple, clear weight displays

### 3. Grid-Based Scanning
- Divides image into grid sections
- Processes each section individually
- Useful for complex images with multiple elements

## Supported Weight Formats

The system can recognize various weight display formats:

- **With Units**: "75.5 kg", "165 lbs", "2500 g"
- **Decimal Values**: "68.2", "150.75"
- **Integer Values**: "70", "180"
- **Different Separators**: Handles both "." and "," as decimal separators

## Output Format

Each detection returns a dictionary with:

```python
{
    'value': 75.5,                    # Numeric weight value
    'unit': 'kg',                     # Weight unit
    'text': '75.5 kg',               # Original OCR text
    'bbox': (x1, y1, x2, y2),       # Bounding box coordinates
    'confidence': 0.95,               # Overall confidence
    'ocr_confidence': 0.92,          # OCR confidence
    'detection_method': 'yolo_display', # Detection method used
    'display_class': 'monitor'        # YOLO detected class (if applicable)
}
```

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Input image or video path | Required |
| `--output` | `-o` | Output path for visualization | Optional |
| `--model` | `-m` | YOLO model path | `yolov8n.pt` |
| `--confidence` | `-c` | Detection confidence threshold | `0.5` |
| `--gpu` | | Use GPU acceleration | `False` |
| `--video` | | Process as video file | `False` |

## Performance Tips

### For Better Accuracy:
1. **High-quality images**: Use clear, well-lit images
2. **Proper resolution**: Ensure text is readable (at least 20px height)
3. **Contrast**: Good contrast between text and background
4. **Stable positioning**: Minimize blur and motion

### For Better Speed:
1. **Use GPU**: Add `--gpu` flag for CUDA acceleration
2. **Lower resolution**: Resize large images before processing
3. **Appropriate model**: Use `yolov8n.pt` for speed, `yolov8x.pt` for accuracy

## Troubleshooting

### Common Issues

1. **No weights detected**:
   - Check image quality and lighting
   - Try adjusting confidence threshold
   - Ensure weight display is clearly visible

2. **Incorrect weight values**:
   - Verify OCR preprocessing is working
   - Check for image distortion or blur
   - Consider manual region selection

3. **Slow processing**:
   - Use GPU acceleration
   - Reduce image resolution
   - Use smaller YOLO model (yolov8n vs yolov8x)

### Error Messages

- **"Could not load image"**: Check file path and format
- **"CUDA not available"**: Install CUDA-compatible PyTorch
- **"OCR error"**: Check EasyOCR installation and dependencies

## Customization

### Adding New Weight Patterns
Modify the `weight_patterns` list in the `WeightDisplayReader` class:

```python
self.weight_patterns = [
    r'(\d+\.?\d*)\s*(?:kg|KG|lbs|LBS|g|G|tons?)',
    r'Weight:\s*(\d+\.?\d*)',  # Custom pattern
    # Add more patterns as needed
]
```

### Custom Display Classes
Update the `display_classes` list to include new device types:

```python
self.display_classes = [
    'tv', 'laptop', 'cell phone', 'monitor', 'screen',
    'scale', 'weighing_machine', 'digital_display'  # Custom classes
]
```

## Examples

### Example 1: Basic Weight Detection
```bash
python yolo.py --input examples/bathroom_scale.jpg
```

### Example 2: Industrial Scale with GPU
```bash
python yolo.py --input examples/industrial_scale.jpg --gpu --output result.jpg
```

### Example 3: Video Processing
```bash
python yolo.py --input examples/weighing_process.mp4 --video --output analyzed_video.mp4
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- **Ultralytics**: For the excellent YOLOv8 implementation
- **EasyOCR**: For robust text recognition capabilities
- **OpenCV**: For computer vision operations
