"""
Simple test script for YOLO Weight Display Reader
Use this to test the weight reader on your own images
"""

import sys
import os
from yolo import WeightDisplayReader

def test_weight_reader():
    """
    Test the weight reader with user input
    """
    print("YOLO Weight Display Reader - Test Script")
    print("=" * 50)
    
    # Get image path from user
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to your weight display image: ").strip()
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return
    
    print(f"Processing image: {image_path}")
    print("-" * 30)
    
    try:
        # Initialize the weight reader
        print("Initializing YOLO Weight Display Reader...")
        reader = WeightDisplayReader(
            confidence_threshold=0.3,
            use_gpu=False  # Set to True if you have CUDA GPU
        )
        
        # Process the image
        print("Analyzing image for weight displays...")
        results = reader.process_image(image_path)
        
        # Display results
        if results:
            print(f"\n✅ SUCCESS! Found {len(results)} weight reading(s):")
            print("=" * 50)
            
            for i, result in enumerate(results, 1):
                print(f"\n📊 Weight Reading #{i}:")
                print(f"   💰 Value: {result['value']:.2f} {result['unit']}")
                print(f"   🔍 Detection Method: {result['detection_method']}")
                print(f"   📍 Position: {result['bbox']}")
                
                if 'ocr_confidence' in result:
                    confidence = result['ocr_confidence']
                    confidence_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                    print(f"   {confidence_emoji} OCR Confidence: {confidence:.2f}")
                
                if 'display_class' in result:
                    print(f"   📱 Display Type: {result['display_class']}")
        else:
            print("\n❌ No weight displays detected in the image.")
            print("\n💡 Tips for better detection:")
            print("   • Ensure the weight display is clearly visible")
            print("   • Good lighting and contrast")
            print("   • Avoid blurry or distorted images")
            print("   • Try different angles if needed")
        
        # Ask if user wants visualization
        save_viz = input(f"\nSave visualization? (y/n): ").lower().strip()
        if save_viz in ['y', 'yes']:
            output_path = image_path.replace('.', '_result.')
            reader.visualize_results(image_path, results, output_path)
            print(f"✅ Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error processing image: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Check if all dependencies are installed: pip install -r requirements.txt")
        print("   • Ensure the image file is not corrupted")
        print("   • Try a different image format (JPG, PNG)")

def main():
    """
    Main function
    """
    print("🔍 YOLO Weight Display Reader - Quick Test")
    print("This script will analyze your image for weight displays\n")
    
    test_weight_reader()
    
    print(f"\n{'='*50}")
    print("📚 For more advanced usage, see:")
    print("   • examples.py - Comprehensive examples")
    print("   • README.md - Full documentation")
    print("   • yolo.py --help - Command line options")

if __name__ == "__main__":
    main()
