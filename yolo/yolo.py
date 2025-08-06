import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
import re
from typing import List, Tuple, Optional, Dict
import argparse
import os
from pathlib import Path
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightDisplayReader:
    """
    YOLO-based weight display reader that can detect and read digital weight displays
    """
    
    def __init__(self, 
                 yolo_model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 use_gpu: bool = True):
        """
        Initialize the weight display reader
        
        Args:
            yolo_model_path: Path to YOLO model weights
            confidence_threshold: Confidence threshold for detections
            use_gpu: Whether to use GPU acceleration
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Initialize YOLO model
        logger.info(f"Loading YOLO model from {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        
        # Initialize OCR reader
        logger.info("Initializing OCR reader")
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        # Define classes that might contain displays (you can customize this)
        self.display_classes = [
            'tv', 'laptop', 'cell phone', 'monitor', 'screen'
        ]
        
        # Weight patterns to look for
        self.weight_patterns = [
            r'(\d+\.?\d*)\s*(?:kg|KG|lbs|LBS|g|G|pounds?)',  # Weight with units
            r'(\d+\.?\d*)',  # Just numbers (assume weight)
        ]
        
    def detect_displays(self, image: np.ndarray) -> List[Dict]:
        """
        Detect potential display areas in the image using YOLO
        
        Args:
            image: Input image
            
        Returns:
            List of detected display regions with bounding boxes
        """
        results = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)
        
        displays = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Check if this could be a display
                    if (class_name.lower() in self.display_classes or 
                        'display' in class_name.lower() or
                        'screen' in class_name.lower()):
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        displays.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'class': class_name,
                            'confidence': confidence
                        })
        
        return displays
    
    def preprocess_display_region(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the display region to enhance OCR accuracy
        
        Args:
            image: Display region image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Invert if background is dark
        if np.mean(cleaned) < 127:
            cleaned = cv2.bitwise_not(cleaned)
        
        return cleaned
    
    def extract_weight_from_text(self, text: str) -> Optional[Dict]:
        """
        Extract weight value from OCR text using regex patterns
        
        Args:
            text: OCR extracted text
            
        Returns:
            Dictionary with weight value and unit, or None if not found
        """
        text = text.strip().replace(',', '.')  
        
        for pattern in self.weight_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                
                if isinstance(matches[0], tuple):
                    weight_str = matches[0][0] if matches[0][0] else matches[0][1]
                else:
                    weight_str = matches[0]
                
                try:
                    weight_value = float(weight_str)
                    
                    # Determine unit
                    unit = 'kg'  # default
                    text_lower = text.lower()
                    if 'lbs' in text_lower or 'pounds' in text_lower:
                        unit = 'lbs'
                    elif 'g' in text_lower and 'kg' not in text_lower:
                        unit = 'g'
                    
                    return {
                        'value': weight_value,
                        'unit': unit,
                        'text': text,
                        'confidence': 1.0
                    }
                except ValueError:
                    continue
        
        return None
    
    def read_weight_from_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Read weight from a specific region of the image
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Weight information or None if not found
        """
        x1, y1, x2, y2 = bbox
        
        # Extract region
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        # Preprocess the region
        processed_region = self.preprocess_display_region(region)
        
        # Perform OCR
        try:
            ocr_results = self.ocr_reader.readtext(processed_region, detail=1)
            
            # Combine all text
            all_text = ' '.join([result[1] for result in ocr_results])
            
            # Extract weight
            weight_info = self.extract_weight_from_text(all_text)
            
            if weight_info:
                # Add OCR confidence
                if ocr_results:
                    avg_confidence = np.mean([result[2] for result in ocr_results])
                    weight_info['ocr_confidence'] = avg_confidence
                
                weight_info['bbox'] = bbox
                
            return weight_info
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None
    
    def process_image(self, image_path: str) -> List[Dict]:
        """
        Process an image to detect and read weight displays
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of detected weights with their information
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return []
        
        results = []
        
        # Method 1: Try to detect displays first
        displays = self.detect_displays(image)
        
        if displays:
            logger.info(f"Found {len(displays)} potential displays")
            for display in displays:
                weight_info = self.read_weight_from_region(image, display['bbox'])
                if weight_info:
                    weight_info['detection_method'] = 'yolo_display'
                    weight_info['display_class'] = display['class']
                    weight_info['display_confidence'] = display['confidence']
                    results.append(weight_info)
        
        # Method 2: If no displays found or no weights in displays, try whole image
        if not results:
            logger.info("No weights found in detected displays, trying whole image")
            height, width = image.shape[:2]
            whole_image_bbox = (0, 0, width, height)
            weight_info = self.read_weight_from_region(image, whole_image_bbox)
            if weight_info:
                weight_info['detection_method'] = 'whole_image'
                results.append(weight_info)
        
        # Method 3: Try grid-based approach for better coverage
        if not results:
            logger.info("Trying grid-based approach")
            results.extend(self.grid_based_detection(image))
        
        return results
    
    def grid_based_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Try a grid-based approach to find weight displays
        
        Args:
            image: Input image
            
        Returns:
            List of detected weights
        """
        height, width = image.shape[:2]
        results = []
        
        # Try different grid sizes
        for grid_rows, grid_cols in [(2, 2), (3, 3), (2, 3)]:
            cell_height = height // grid_rows
            cell_width = width // grid_cols
            
            for row in range(grid_rows):
                for col in range(grid_cols):
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = min((col + 1) * cell_width, width)
                    y2 = min((row + 1) * cell_height, height)
                    
                    weight_info = self.read_weight_from_region(image, (x1, y1, x2, y2))
                    if weight_info:
                        weight_info['detection_method'] = f'grid_{grid_rows}x{grid_cols}'
                        results.append(weight_info)
        
        return results
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video to detect and read weight displays
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame to reduce computation
            if frame_count % 10 == 0:
                # Save frame temporarily
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Process frame
                weights = self.process_image(temp_path)
                
                # Draw results on frame
                for weight in weights:
                    bbox = weight['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw weight text
                    weight_text = f"{weight['value']:.1f} {weight['unit']}"
                    cv2.putText(frame, weight_text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if output_path:
                out.write(frame)
            
            # Display frame (optional)
            cv2.imshow('Weight Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def visualize_results(self, image_path: str, results: List[Dict], save_path: Optional[str] = None):
        """
        Visualize detection results on the image
        
        Args:
            image_path: Path to input image
            results: Detection results
            save_path: Path to save visualization (optional)
        """
        image = cv2.imread(image_path)
        
        for i, result in enumerate(results):
            bbox = result['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for weight detection
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Prepare text
            weight_text = f"{result['value']:.1f} {result['unit']}"
            method_text = f"Method: {result['detection_method']}"
            
            # Draw text background
            text_size = cv2.getTextSize(weight_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(image, (x1, y1-60), (x1 + text_size[0] + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(image, weight_text, (x1+5, y1-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, method_text, (x1+5, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, image)
            logger.info(f"Visualization saved to {save_path}")
        
        return image


def main():
    """
    Main function to run the weight display reader
    """
    parser = argparse.ArgumentParser(description="YOLO Weight Display Reader")
    parser.add_argument("--input", "-i", required=True, help="Input image or video path")
    parser.add_argument("--output", "-o", help="Output path for visualization")
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--video", action="store_true", help="Process as video")
    
    args = parser.parse_args()
    
    # Initialize reader
    reader = WeightDisplayReader(
        yolo_model_path=args.model,
        confidence_threshold=args.confidence,
        use_gpu=args.gpu
    )
    
    if args.video:
        # Process video
        reader.process_video(args.input, args.output)
    else:
        # Process image
        start_time = time.time()
        results = reader.process_image(args.input)
        processing_time = time.time() - start_time
        
        # Print results
        if results:
            print("\n" + "="*50)
            print("WEIGHT DETECTION RESULTS")
            print("="*50)
            for i, result in enumerate(results, 1):
                print(f"\nDetection {i}:")
                print(f"  Weight: {result['value']:.2f} {result['unit']}")
                print(f"  Method: {result['detection_method']}")
                print(f"  Bounding Box: {result['bbox']}")
                if 'ocr_confidence' in result:
                    print(f"  OCR Confidence: {result['ocr_confidence']:.2f}")
                if 'display_class' in result:
                    print(f"  Display Type: {result['display_class']}")
        else:
            print("No weight displays detected in the image.")
        
        print(f"\nProcessing time: {processing_time:.2f} seconds")
        
        # Create visualization
        if args.output:
            reader.visualize_results(args.input, results, args.output)
        else:
            # Show visualization
            vis_image = reader.visualize_results(args.input, results)
            cv2.imshow("Weight Detection Results", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()