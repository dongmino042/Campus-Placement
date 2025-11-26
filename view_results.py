#!/usr/bin/env python3
"""
view_results.py

Simple script to view the results after running run_models.py

Usage:
  python view_results.py
"""
import json
import os

def view_results(results_dir='outputs'):
    """Display results from the model training."""
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"❌ Results directory '{results_dir}' not found.")
        print(f"Please run 'python run_models.py' first to generate results.")
        return
    
    # Check for summary file
    summary_file = os.path.join(results_dir, 'results_summary.txt')
    if os.path.exists(summary_file):
        print("\n" + "="*80)
        print("📄 RESULTS SUMMARY")
        print("="*80)
        with open(summary_file, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print(f"❌ Summary file not found: {summary_file}")
    
    # Check for JSON results
    json_file = os.path.join(results_dir, 'results.json')
    if os.path.exists(json_file):
        print("\n" + "="*80)
        print("📋 RESULTS JSON (Programmatic Access)")
        print("="*80)
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"\n🔍 To use results in your code:")
        print(f"```python")
        print(f"import json")
        print(f"")
        print(f"with open('{json_file}', 'r') as f:")
        print(f"    results = json.load(f)")
        print(f"")
        print(f"best_clf_model = results['best_classification_model']")
        print(f"best_reg_model = results['best_regression_model']")
        print(f"accuracy = results['classification_results'][best_clf_model]['accuracy']")
        print(f"```")
    
    # List all output files
    print("\n" + "="*80)
    print("📁 OUTPUT FILES")
    print("="*80)
    if os.path.exists(results_dir):
        files = sorted(os.listdir(results_dir))
        for file in files:
            filepath = os.path.join(results_dir, file)
            size = os.path.getsize(filepath)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
            
            # Add emoji based on file type
            if file.endswith('.png'):
                emoji = "📊"
            elif file.endswith('.json'):
                emoji = "📋"
            elif file.endswith('.txt'):
                emoji = "📄"
            elif file.endswith('.joblib'):
                emoji = "💾"
            else:
                emoji = "📁"
            
            print(f"  {emoji} {file:50s} {size_str:>10s}")
    
    print("\n" + "="*80)
    print("✅ View complete! Check the outputs directory for all files.")
    print("="*80 + "\n")

if __name__ == "__main__":
    view_results()
