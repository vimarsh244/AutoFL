#!/usr/bin/env python3
"""
Script to analyze the weird OMNeT++ CSV files and understand their structure.
"""

import csv
import pandas as pd
import numpy as np
import os

def analyze_csv_file(filepath):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*80}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    # Try to get basic file info
    try:
        file_size = os.path.getsize(filepath)
        print(f"📁 File size: {file_size:,} bytes")
    except Exception as e:
        print(f"❌ Error getting file size: {e}")
    
    # First, let's try to read with standard CSV to see structure
    print(f"\n🔍 BASIC CSV STRUCTURE:")
    try:
        with open(filepath, 'r') as f:
            # Read first 20 lines to understand structure
            lines = []
            for i, line in enumerate(f):
                if i < 20:
                    lines.append(line.strip())
                else:
                    break
            
            print(f"📊 First 20 lines:")
            for i, line in enumerate(lines, 1):
                print(f"  {i:2d}: {line[:100]}{'...' if len(line) > 100 else ''}")
    except Exception as e:
        print(f"❌ Error reading basic structure: {e}")
    
    # Try to understand CSV structure with pandas
    print(f"\n🐼 PANDAS ANALYSIS:")
    try:
        # Try different separators
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(filepath, sep=sep, nrows=100)
                if len(df.columns) > 1:
                    print(f"✅ Successfully read with separator '{sep}'")
                    print(f"📋 Columns ({len(df.columns)}): {list(df.columns)}")
                    print(f"📏 Shape (first 100 rows): {df.shape}")
                    
                    # Show data types
                    print(f"🔢 Data types:")
                    for col, dtype in df.dtypes.items():
                        print(f"   {col}: {dtype}")
                    
                    # Show first few rows
                    print(f"📄 First 5 rows:")
                    print(df.head().to_string())
                    
                    # Look for vector data specifically
                    if 'vectime' in df.columns and 'vecvalue' in df.columns:
                        print(f"\n🎯 VECTOR DATA ANALYSIS:")
                        vector_rows = df[df['vectime'].notna() & df['vecvalue'].notna()]
                        print(f"📈 Vector rows found: {len(vector_rows)}")
                        
                        if len(vector_rows) > 0:
                            print(f"🔍 Sample vector rows:")
                            print(vector_rows.head(3).to_string())
                            
                            # Analyze vector content
                            for idx, row in vector_rows.head(3).iterrows():
                                print(f"\n   Vector {idx}:")
                                print(f"     Module: {row.get('module', 'N/A')}")
                                print(f"     Name: {row.get('name', 'N/A')}")
                                print(f"     VecTime length: {len(str(row['vectime']))}")
                                print(f"     VecValue length: {len(str(row['vecvalue']))}")
                                
                                # Try to parse the vector data
                                try:
                                    times = str(row['vectime']).split()
                                    values = str(row['vecvalue']).split()
                                    print(f"     Time samples: {len(times)}")
                                    print(f"     Value samples: {len(values)}")
                                    if len(times) > 0:
                                        print(f"     Time range: {times[0]} to {times[-1] if len(times) > 1 else times[0]}")
                                    if len(values) > 0:
                                        print(f"     Value range: {values[0]} to {values[-1] if len(values) > 1 else values[0]}")
                                except Exception as e:
                                    print(f"     ❌ Error parsing vector: {e}")
                    
                    break
            except Exception as e:
                continue
        else:
            print(f"❌ Could not read with any common separator")
    except Exception as e:
        print(f"❌ Error with pandas analysis: {e}")
    
    # Try manual parsing focusing on the weird structure
    print(f"\n🔧 MANUAL PARSING ANALYSIS:")
    try:
        with open(filepath, 'r') as f:
            all_lines = f.readlines()
        
        print(f"📊 Total lines: {len(all_lines)}")
        
        # Look for patterns
        vector_lines = []
        attr_lines = []
        other_lines = []
        
        for i, line in enumerate(all_lines):
            line = line.strip()
            if 'vector' in line:
                vector_lines.append((i+1, line))
            elif 'attr' in line:
                attr_lines.append((i+1, line))
            elif line and not line.startswith('#'):
                other_lines.append((i+1, line))
        
        print(f"🎯 Vector lines: {len(vector_lines)}")
        print(f"⚙️  Attribute lines: {len(attr_lines)}")
        print(f"📄 Other content lines: {len(other_lines)}")
        
        # Show some examples
        if vector_lines:
            print(f"\n🎯 Sample vector lines:")
            for line_num, line in vector_lines[:3]:
                print(f"   Line {line_num}: {line[:200]}{'...' if len(line) > 200 else ''}")
        
        if attr_lines:
            print(f"\n⚙️  Sample attribute lines:")
            for line_num, line in attr_lines[:3]:
                print(f"   Line {line_num}: {line[:200]}{'...' if len(line) > 200 else ''}")
        
        # Look for the actual data around line 94 as mentioned
        print(f"\n🔍 AROUND LINE 94 (as user mentioned):")
        start_line = max(0, 90)
        end_line = min(len(all_lines), 100)
        for i in range(start_line, end_line):
            line = all_lines[i].strip()
            if line:
                print(f"   Line {i+1}: {line[:200]}{'...' if len(line) > 200 else ''}")
    
    except Exception as e:
        print(f"❌ Error with manual parsing: {e}")

def main():
    print("🚀 OMNeT++ CSV Analysis Tool")
    print("="*80)
    
    csv_files = [
        "Cars2BS_Sim_Data/Car2BS_Mean_SD.csv",
        "Cars2BS_Sim_Data/Car2BS_Multi_Mean_SD.csv", 
        "Cars2BS_Sim_Data/BS2Car.csv"
    ]
    
    for csv_file in csv_files:
        analyze_csv_file(csv_file)
    
    print(f"\n{'='*80}")
    print("🎯 SUMMARY AND RECOMMENDATIONS:")
    print("="*80)
    print("Based on the analysis above, we can see:")
    print("1. The CSV structure and where the actual vector data is located")
    print("2. What signals are available and their naming patterns")
    print("3. How many cars/nodes we actually have data for")
    print("4. The time ranges and transmission patterns")
    print("\nNext steps:")
    print("- Update the sim/availability.py parser based on findings")
    print("- Adjust configuration to match actual data structure")
    print("- Ensure we have enough cars for 10 FL clients")

if __name__ == "__main__":
    main()
