#!/usr/bin/env python3
"""
Test script to verify the dashboard can find and process result directories
"""

from dashboard import ResultsPlotter
from pathlib import Path

def test_directory_discovery():
    """Test if the plotter can find result directories"""
    plotter = ResultsPlotter()
    
    print("🔍 Searching for result directories...")
    directories = plotter.find_result_directories()
    
    print(f"📁 Found {len(directories)} directories with results:")
    for i, directory in enumerate(directories[:10]):  # Show first 10
        rel_path = directory.relative_to(plotter.base_path)
        data_format = plotter.detect_data_format(directory)
        print(f"  {i+1:2d}. {rel_path} ({data_format})")
    
    if len(directories) > 10:
        print(f"     ... and {len(directories) - 10} more")
    
    return directories

def test_data_processing():
    """Test if the plotter can process different data formats"""
    plotter = ResultsPlotter()
    directories = plotter.find_result_directories()
    
    print(f"\n🧪 Testing data processing on first few directories...")
    
    for directory in directories[:5]:
        rel_path = directory.relative_to(plotter.base_path)
        data_format = plotter.detect_data_format(directory)
        
        print(f"\n📊 Processing: {rel_path}")
        print(f"   Format: {data_format}")
        
        data = plotter.process_directory(directory)
        if data:
            times = data["times"]
            values = data["values"]
            print(f"   ✅ Success: {len(times)} time points, values range [{min(values):.4f}, {max(values):.4f}]")
            if "runs" in data:
                print(f"   📈 Averaged across {data['runs']} runs")
            if "method" in data:
                print(f"   🎯 Method: {data['method']}")
        else:
            print(f"   ❌ Failed to process")

if __name__ == "__main__":
    print("🚀 Testing Optimization Results Dashboard")
    print("=" * 50)
    
    # Test directory discovery
    directories = test_directory_discovery()
    
    if directories:
        # Test data processing
        test_data_processing()
        
        print(f"\n✅ Dashboard should work! Found {len(directories)} processable directories.")
        print("💡 Run 'python dashboard.py' to start the web interface")
    else:
        print("\n❌ No result directories found. Check your WIP folder structure.")
    
    print("\n🎯 Ready to run dashboard!") 