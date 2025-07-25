#!/usr/bin/env python3
"""
Check Python version compatibility for the OCR Evaluation Framework.
"""

import sys
import platform


def check_python_version():
    """Check if the current Python version is compatible."""
    current_version = sys.version_info
    min_version = (3, 9)
    recommended_version = (3, 13)
    
    print("🐍 Python Version Check")
    print("=" * 30)
    print(f"Current Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # Check minimum version
    if current_version >= min_version:
        print(f"✅ Compatible with minimum version {min_version[0]}.{min_version[1]}+")
    else:
        print(f"❌ Requires Python {min_version[0]}.{min_version[1]}+")
        return False
    
    # Check recommended version
    if current_version >= recommended_version:
        print(f"🎉 Using recommended version {recommended_version[0]}.{recommended_version[1]}+")
        print("   You'll have access to the latest Python features and optimizations!")
    else:
        print(f"⚠️  Recommended: Python {recommended_version[0]}.{recommended_version[1]}+")
        print("   Consider upgrading for better performance and latest features.")
    
    # Check for specific Python 3.13+ features
    if current_version >= (3, 13):
        print("\n🚀 Python 3.13+ Features Available:")
        print("   - Improved error messages")
        print("   - Better performance")
        print("   - Enhanced type checking")
        print("   - Latest security updates")
    
    return True


def check_dependencies():
    """Check if key dependencies are available."""
    print("\n📦 Dependency Check")
    print("=" * 30)
    
    dependencies = [
        ("numpy", "Numerical computing"),
        ("cv2", "OpenCV for image processing"),
        ("PIL", "Pillow for image handling"),
        ("pandas", "Data manipulation"),
        ("matplotlib", "Plotting and visualization"),
        ("sklearn", "Machine learning utilities"),
    ]
    
    for dep_name, description in dependencies:
        try:
            if dep_name == "cv2":
                import cv2
                version = cv2.__version__
            elif dep_name == "PIL":
                import PIL
                version = PIL.__version__
            elif dep_name == "sklearn":
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(dep_name)
                version = getattr(module, "__version__", "unknown")
            
            print(f"✅ {dep_name} ({description}): {version}")
        except ImportError:
            print(f"❌ {dep_name} ({description}): Not installed")
    
    print("\n💡 To install dependencies:")
    print("   pipenv install")


def main():
    """Main function."""
    print("OCR Evaluation Framework - Python Compatibility Check")
    print("=" * 55)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    check_dependencies()
    
    print("\n🎯 Framework Status:")
    print("   - Python version: ✅ Compatible")
    print("   - Dependencies: Install with 'pipenv install'")
    print("   - Ready to use: Run './quick_start.sh' or 'make install'")


if __name__ == "__main__":
    main() 