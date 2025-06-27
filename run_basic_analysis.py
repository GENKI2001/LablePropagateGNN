#!/usr/bin/env python3
"""
Script to run basic dataset analysis
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_basic_analyzer import DatasetBasicAnalyzer

def main():
    """
    Main function to run basic dataset analysis
    """
    print("=== Dataset Basic Information Analysis ===")
    
    # Initialize analyzer
    analyzer = DatasetBasicAnalyzer()
    
    # Analyze all datasets
    analyzer.analyze_all_datasets(save_plots=True, output_dir='./')
    
    print("\n=== Analysis Complete ===")
    print("Images saved in 'dataset_basic_images' folder")

if __name__ == "__main__":
    main() 