"""
System Test Script
Verifies that all components are properly installed and configured
"""

import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_test(name, passed, message=""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")
    if message:
        print(f"       {message}")

def test_python_version():
    """Test Python version"""
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 8
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    print_test("Python Version (>=3.8)", passed, message)
    return passed

def test_imports():
    """Test required package imports"""
    packages = {
        'flask': 'Flask',
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'PIL': 'Pillow',
        'efficientnet_pytorch': 'EfficientNet',
        'piexif': 'Piexif',
        'sklearn': 'scikit-learn'
    }
    
    results = []
    for package, name in packages.items():
        try:
            __import__(package)
            print_test(f"Import {name}", True)
            results.append(True)
        except ImportError:
            print_test(f"Import {name}", False, f"Run: pip install {package}")
            results.append(False)
    
    return all(results)

def test_directory_structure():
    """Test directory structure"""
    required_dirs = [
        '../backend',
        '../frontend',
        '../models',
        '../datasets',
        '../datasets/train',
        '../static/uploads'
    ]
    
    results = []
    for directory in required_dirs:
        exists = os.path.exists(directory)
        print_test(f"Directory: {directory}", exists)
        results.append(exists)
    
    return all(results)

def test_backend_files():
    """Test backend files exist"""
    required_files = [
        '../backend/app.py',
        '../backend/train_model.py',
        '../backend/frequency_analyzer.py',
        '../backend/noise_analyzer.py',
        '../backend/metadata_analyzer.py',
        '../backend/pixel_analyzer.py'
    ]
    
    results = []
    for filepath in required_files:
        exists = os.path.exists(filepath)
        print_test(f"File: {os.path.basename(filepath)}", exists)
        results.append(exists)
    
    return all(results)

def test_frontend_files():
    """Test frontend files exist"""
    exists = os.path.exists('../frontend/index.html')
    print_test("Frontend: index.html", exists)
    return exists

def test_model_exists():
    """Test if trained model exists"""
    model_paths = [
        '../models/best_model.pth',
        '../models/final_model.pth'
    ]
    
    found = False
    for path in model_paths:
        if os.path.exists(path):
            print_test(f"Trained Model: {os.path.basename(path)}", True)
            found = True
    
    if not found:
        print_test("Trained Model", False, "Run train_model.py first")
    
    return found

def test_dataset():
    """Test if dataset is present"""
    train_dirs = [
        '../datasets/train/real',
        '../datasets/train/stable_diffusion'
    ]
    
    has_data = False
    for directory in train_dirs:
        if os.path.exists(directory):
            file_count = len([f for f in os.listdir(directory) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if file_count > 0:
                print_test(f"Dataset: {os.path.basename(directory)}", True, 
                          f"{file_count} images")
                has_data = True
            else:
                print_test(f"Dataset: {os.path.basename(directory)}", False, 
                          "No images found")
    
    if not has_data:
        print_test("Training Data", False, 
                  "Run datasets/download_datasets.py or add images manually")
    
    return has_data

def run_quick_analysis_test():
    """Test analysis modules with a dummy image"""
    try:
        import numpy as np
        import cv2
        sys.path.append('../backend')
        
        from frequency_analyzer import FrequencyAnalyzer
        from noise_analyzer import NoiseAnalyzer
        from pixel_analyzer import PixelStatisticsAnalyzer
        
        # Create a test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test frequency analyzer
        freq_analyzer = FrequencyAnalyzer()
        freq_result = freq_analyzer.analyze(test_image)
        print_test("Frequency Analyzer", 'frequency_score' in freq_result)
        
        # Test noise analyzer
        noise_analyzer = NoiseAnalyzer()
        noise_result = noise_analyzer.analyze(test_image)
        print_test("Noise Analyzer", 'noise_score' in noise_result)
        
        # Test pixel analyzer
        pixel_analyzer = PixelStatisticsAnalyzer()
        pixel_result = pixel_analyzer.analyze(test_image)
        print_test("Pixel Analyzer", 'pixel_score' in pixel_result)
        
        return True
        
    except Exception as e:
        print_test("Analysis Modules", False, str(e))
        return False

def print_recommendations(results):
    """Print recommendations based on test results"""
    print_header("RECOMMENDATIONS")
    
    if not results['python']:
        print("⚠️  Upgrade to Python 3.8 or higher")
    
    if not results['imports']:
        print("⚠️  Install missing packages: pip install -r requirements.txt")
    
    if not results['dataset']:
        print("⚠️  Set up training data:")
        print("    1. Run: cd datasets && python download_datasets.py")
        print("    2. Or manually add images to datasets/train/")
    
    if not results['model']:
        print("⚠️  Train the model:")
        print("    1. cd backend")
        print("    2. python train_model.py")
    
    if all([results['python'], results['imports'], results['directories'], 
            results['backend'], results['frontend']]):
        print("✅ System components are properly installed!")
        
        if results['dataset'] and results['model']:
            print("✅ Ready to run!")
            print("\nNext steps:")
            print("  1. cd backend && python app.py")
            print("  2. Open frontend/index.html in browser")
        elif results['dataset'] and not results['model']:
            print("✅ Dataset ready!")
            print("\nNext step: Train the model")
            print("  cd backend && python train_model.py")
        else:
            print("\nNext step: Set up dataset")
            print("  cd datasets && python download_datasets.py")

def main():
    """Main test function"""
    print_header("AI IMAGE DETECTOR - SYSTEM TEST")
    
    results = {}
    
    print_header("1. PYTHON VERSION")
    results['python'] = test_python_version()
    
    print_header("2. REQUIRED PACKAGES")
    results['imports'] = test_imports()
    
    print_header("3. DIRECTORY STRUCTURE")
    results['directories'] = test_directory_structure()
    
    print_header("4. BACKEND FILES")
    results['backend'] = test_backend_files()
    
    print_header("5. FRONTEND FILES")
    results['frontend'] = test_frontend_files()
    
    print_header("6. TRAINING DATA")
    results['dataset'] = test_dataset()
    
    print_header("7. TRAINED MODEL")
    results['model'] = test_model_exists()
    
    print_header("8. ANALYSIS MODULES")
    results['analysis'] = run_quick_analysis_test()
    
    # Print summary
    print_header("SUMMARY")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! System is ready to use!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
    
    # Print recommendations
    print_recommendations(results)
    
    print("\n" + "="*70)
    print("For detailed instructions, see:")
    print("  - README.md (comprehensive guide)")
    print("  - QUICKSTART.md (5-step quick start)")
    print("="*70 + "\n")

if __name__ == '__main__':
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
