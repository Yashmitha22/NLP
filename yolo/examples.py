"""
Example script demonstrating YOLO Weight Display Reader usage
"""

import cv2
import numpy as np
from yolo import WeightDisplayReader
import os
import time

def create_sample_weight_display():
    """
    Create a sample weight display image for testing
    """
    # Create a simple digital display simulation
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Create display background (dark gray)
    cv2.rectangle(img, (50, 100), (550, 300), (40, 40, 40), -1)
    
    # Create display border
    cv2.rectangle(img, (50, 100), (550, 300), (200, 200, 200), 3)
    
    # Add weight text
    font = cv2.FONT_HERSHEY_SIMPLEX
    weight_text = "75.5 kg"
    text_size = cv2.getTextSize(weight_text, font, 3, 3)[0]
    text_x = (600 - text_size[0]) // 2
    text_y = (400 + text_size[1]) // 2
    
    cv2.putText(img, weight_text, (text_x, text_y), font, 3, (0, 255, 0), 3)
    
    # Add some additional elements to make it more realistic
    cv2.putText(img, "DIGITAL SCALE", (200, 80), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "STABLE", (480, 350), font, 0.6, (0, 255, 0), 2)
    
    return img

def example_1_basic_usage():
    """
    Example 1: Basic weight detection from an image
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Weight Detection")
    print("="*60)
    
    # Create sample image
    sample_img = create_sample_weight_display()
    sample_path = "sample_weight_display.jpg"
    cv2.imwrite(sample_path, sample_img)
    print(f"Created sample image: {sample_path}")
    
    # Initialize reader
    print("Initializing YOLO Weight Display Reader...")
    reader = WeightDisplayReader(
        confidence_threshold=0.3,
        use_gpu=False  # Set to True if you have CUDA
    )
    
    # Process the image
    print("Processing image...")
    start_time = time.time()
    results = reader.process_image(sample_path)
    processing_time = time.time() - start_time
    
    # Display results
    if results:
        print(f"\n✓ Found {len(results)} weight reading(s):")
        for i, result in enumerate(results, 1):
            print(f"\n  Reading {i}:")
            print(f"    Weight: {result['value']:.2f} {result['unit']}")
            print(f"    Detection Method: {result['detection_method']}")
            print(f"    Bounding Box: {result['bbox']}")
            if 'ocr_confidence' in result:
                print(f"    OCR Confidence: {result['ocr_confidence']:.2f}")
    else:
        print("\n✗ No weight readings detected")
    
    print(f"\nProcessing time: {processing_time:.2f} seconds")
    
    # Create visualization
    output_path = "result_example1.jpg"
    reader.visualize_results(sample_path, results, output_path)
    print(f"Visualization saved: {output_path}")
    
    # Clean up
    if os.path.exists(sample_path):
        os.remove(sample_path)

def example_2_api_usage():
    """
    Example 2: Using the API programmatically
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Programmatic API Usage")
    print("="*60)
    
    # Create a more complex sample
    img = np.zeros((500, 800, 3), dtype=np.uint8)
    
    # Add multiple weight displays
    displays = [
        {"pos": (50, 50, 350, 200), "weight": "68.2 kg", "color": (0, 255, 0)},
        {"pos": (450, 50, 750, 200), "weight": "150 lbs", "color": (0, 255, 255)},
        {"pos": (200, 300, 600, 450), "weight": "2.5 tons", "color": (255, 255, 0)}
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for display in displays:
        x1, y1, x2, y2 = display["pos"]
        
        # Draw display background
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
        
        # Add weight text
        text_size = cv2.getTextSize(display["weight"], font, 1.5, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        
        cv2.putText(img, display["weight"], (text_x, text_y), 
                   font, 1.5, display["color"], 2)
    
    sample_path = "multi_display_sample.jpg"
    cv2.imwrite(sample_path, img)
    
    # Process with API
    reader = WeightDisplayReader(confidence_threshold=0.2, use_gpu=False)
    results = reader.process_image(sample_path)
    
    print(f"Detected {len(results)} weight readings:")
    
    total_weight_kg = 0
    for i, result in enumerate(results, 1):
        weight_kg = result['value']
        unit = result['unit']
        
        # Convert to kg for total calculation
        if unit == 'lbs':
            weight_kg *= 0.453592
        elif unit == 'g':
            weight_kg /= 1000
        elif unit == 'tons':
            weight_kg *= 1000
        
        total_weight_kg += weight_kg
        
        print(f"\n  Display {i}: {result['value']:.1f} {unit}")
        print(f"    Method: {result['detection_method']}")
        print(f"    Region: {result['bbox']}")
    
    print(f"\nTotal weight: {total_weight_kg:.2f} kg")
    
    # Visualize
    output_path = "result_example2.jpg"
    reader.visualize_results(sample_path, results, output_path)
    print(f"Visualization saved: {output_path}")
    
    # Clean up
    if os.path.exists(sample_path):
        os.remove(sample_path)

def example_3_custom_preprocessing():
    """
    Example 3: Custom preprocessing for challenging images
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Preprocessing")
    print("="*60)
    
    # Create a challenging image with noise and poor contrast
    img = np.random.randint(40, 80, (400, 600, 3), dtype=np.uint8)  # Noisy background
    
    # Add a low-contrast display
    cv2.rectangle(img, (100, 150), (500, 250), (60, 60, 60), -1)
    cv2.rectangle(img, (100, 150), (500, 250), (100, 100, 100), 2)
    
    # Add barely visible text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "42.7 kg", (200, 210), font, 2, (120, 120, 120), 2)
    
    sample_path = "challenging_sample.jpg"
    cv2.imwrite(sample_path, img)
    
    # Custom reader with adjusted parameters
    class CustomWeightReader(WeightDisplayReader):
        def preprocess_display_region(self, image):
            """Enhanced preprocessing for challenging images"""
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply stronger denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Apply multiple thresholding techniques and combine
            thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Combine thresholds
            combined = cv2.bitwise_or(thresh1, thresh2)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
    
    # Use custom reader
    reader = CustomWeightReader(confidence_threshold=0.1, use_gpu=False)
    results = reader.process_image(sample_path)
    
    if results:
        print("✓ Successfully detected weight despite challenging conditions:")
        for result in results:
            print(f"  Weight: {result['value']:.1f} {result['unit']}")
            print(f"  Method: {result['detection_method']}")
    else:
        print("✗ No weights detected - image too challenging")
    
    # Visualize
    output_path = "result_example3.jpg"
    reader.visualize_results(sample_path, results, output_path)
    print(f"Visualization saved: {output_path}")
    
    # Clean up
    if os.path.exists(sample_path):
        os.remove(sample_path)

def main():
    """
    Run all examples
    """
    print("YOLO Weight Display Reader - Examples")
    print("This script demonstrates various usage scenarios")
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_api_usage()
        example_3_custom_preprocessing()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- result_example1.jpg: Basic detection visualization")
        print("- result_example2.jpg: Multi-display detection")
        print("- result_example3.jpg: Enhanced preprocessing result")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have installed all dependencies:")
        print("  python setup.py")
        print("or")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
