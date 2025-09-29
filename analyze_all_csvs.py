#!/usr/bin/env python3
"""
Comprehensive analysis of all OMNeT++ CSV files to understand their weird structure
"""

import csv
import os
import pandas as pd
from pathlib import Path

def analyze_csv_file(csv_path):
    """Analyze a single CSV file and extract key information"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {csv_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(csv_path):
        print(f"❌ File does not exist: {csv_path}")
        return
    
    try:
        # First, let's see the raw structure
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        print(f"📊 File size: {len(lines)} lines")
        print(f"📊 File size (bytes): {os.path.getsize(csv_path):,}")
        
        # Check if it has vectime/vecvalue columns (indicates vector data)
        first_line = lines[0].strip()
        has_vector_cols = 'vectime' in first_line and 'vecvalue' in first_line
        print(f"📊 Has vector columns: {has_vector_cols}")
        
        # Parse with pandas to understand structure
        try:
            df = pd.read_csv(csv_path)
            print(f"📊 Columns: {list(df.columns)}")
            print(f"📊 Shape: {df.shape}")
            
            # Look for different types of data
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                print(f"📊 Data types:")
                for dtype, count in type_counts.items():
                    print(f"    {dtype}: {count}")
            
            # Look for vector data specifically
            if has_vector_cols:
                vector_data = df[df['type'] == 'vector']
                print(f"📊 Vector rows: {len(vector_data)}")
                
                if len(vector_data) > 0:
                    print(f"📊 Vector modules:")
                    for idx, row in vector_data.iterrows():
                        module = row.get('module', 'N/A')
                        name = row.get('name', 'N/A')
                        vectime = str(row.get('vectime', ''))[:100]  # First 100 chars
                        vecvalue = str(row.get('vecvalue', ''))[:100]  # First 100 chars
                        
                        print(f"    Vector {idx}: module='{module}', name='{name}'")
                        print(f"        vectime preview: {vectime}...")
                        print(f"        vecvalue preview: {vecvalue}...")
                        
                        # Count actual data points
                        if pd.notna(row.get('vectime')):
                            time_points = str(row['vectime']).split()
                            value_points = str(row['vecvalue']).split()
                            print(f"        Data points: {len(time_points)} times, {len(value_points)} values")
            
            # Look for other interesting patterns
            if 'module' in df.columns:
                unique_modules = df['module'].dropna().unique()
                print(f"📊 Unique modules ({len(unique_modules)}):")
                for module in sorted(unique_modules)[:10]:  # Show first 10
                    if module and 'car' in module.lower():
                        print(f"    🚗 {module}")
                    elif module and any(term in module.lower() for term in ['highway', 'upf', 'router', 'gnodeb']):
                        print(f"    🏗️  {module}")
                    elif module:
                        print(f"    📡 {module}")
                
                if len(unique_modules) > 10:
                    print(f"    ... and {len(unique_modules) - 10} more")
            
            # Look for signal names
            if 'name' in df.columns:
                unique_names = df['name'].dropna().unique()
                signal_names = [name for name in unique_names if any(term in name.lower() for term in ['transmission', 'rx', 'tx', 'sinr', 'stat'])]
                print(f"📊 Signal names of interest:")
                for name in sorted(signal_names):
                    count = len(df[df['name'] == name])
                    print(f"    📡 {name}: {count} occurrences")
            
        except Exception as e:
            print(f"❌ Error parsing with pandas: {e}")
            
        # Show first few lines for manual inspection
        print(f"\n📋 First 5 lines:")
        for i, line in enumerate(lines[:5]):
            print(f"    Line {i+1}: {line.strip()}")
        
        # Show last few lines to see if there's data at the end
        print(f"\n📋 Last 5 lines:")
        for i, line in enumerate(lines[-5:], len(lines)-4):
            print(f"    Line {i}: {line.strip()}")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")

def find_transmission_data():
    """Look for transmission-related data across all files"""
    print(f"\n{'='*80}")
    print("SEARCHING FOR TRANSMISSION DATA ACROSS ALL FILES")
    print(f"{'='*80}")
    
    csv_files = [
        "Cars2BS_Sim_Data/Car2BS_Mean_SD.csv",
        "Cars2BS_Sim_Data/Car2BS_Multi_Mean_SD.csv", 
        "Cars2BS_Sim_Data/BS2Car.csv",
        "Cars2BS_Sim_Data/BS2Car_Mean_SD.csv"
    ]
    
    all_transmission_data = []
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"⚠️  Skipping {csv_file} - file not found")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            
            # Look for transmission-related vectors
            if 'type' in df.columns and 'name' in df.columns:
                vector_data = df[df['type'] == 'vector']
                transmission_vectors = vector_data[
                    vector_data['name'].str.contains('transmission|tx|rx|sinr', case=False, na=False)
                ]
                
                if len(transmission_vectors) > 0:
                    print(f"\n📁 {csv_file}: Found {len(transmission_vectors)} transmission vectors")
                    for idx, row in transmission_vectors.iterrows():
                        module = row.get('module', 'N/A')
                        name = row.get('name', 'N/A')
                        
                        # Extract car index from module name if possible
                        car_idx = "unknown"
                        if 'car[' in module:
                            try:
                                car_idx = module.split('car[')[1].split(']')[0]
                            except:
                                pass
                        
                        # Count data points
                        data_points = 0
                        if pd.notna(row.get('vectime')):
                            data_points = len(str(row['vectime']).split())
                        
                        print(f"    🚗 Car {car_idx}: {name} ({data_points} data points)")
                        print(f"        Module: {module}")
                        
                        all_transmission_data.append({
                            'file': csv_file,
                            'car_idx': car_idx,
                            'module': module,
                            'signal': name,
                            'data_points': data_points,
                            'vectime': row.get('vectime'),
                            'vecvalue': row.get('vecvalue')
                        })
                        
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
    
    print(f"\n📊 SUMMARY: Found {len(all_transmission_data)} transmission vectors total")
    
    # Group by car to see what we have
    cars_found = {}
    for data in all_transmission_data:
        car_idx = data['car_idx']
        if car_idx not in cars_found:
            cars_found[car_idx] = []
        cars_found[car_idx].append(data)
    
    print(f"📊 Cars with transmission data: {len(cars_found)}")
    for car_idx, car_data in cars_found.items():
        print(f"    🚗 Car {car_idx}: {len(car_data)} signals")
        for signal_data in car_data:
            print(f"        📡 {signal_data['signal']} ({signal_data['data_points']} points) from {signal_data['file']}")
    
    return all_transmission_data

def main():
    """Main analysis function"""
    print("🔍 ANALYZING ALL OMNET++ CSV FILES")
    
    # List of CSV files to analyze
    csv_files = [
        "Cars2BS_Sim_Data/Car2BS_Mean_SD.csv",
        "Cars2BS_Sim_Data/Car2BS_Multi_Mean_SD.csv", 
        "Cars2BS_Sim_Data/BS2Car.csv",
        "Cars2BS_Sim_Data/BS2Car_Mean_SD.csv"
    ]
    
    # Analyze each file individually
    for csv_file in csv_files:
        analyze_csv_file(csv_file)
    
    # Look for transmission data across all files
    transmission_data = find_transmission_data()
    
    # Recommendations
    print(f"\n{'='*80}")
    print("📋 RECOMMENDATIONS FOR FL INTEGRATION")
    print(f"{'='*80}")
    
    if len(transmission_data) >= 10:
        print("✅ Good: Found enough transmission data for 10 FL clients")
        print("💡 Recommendation: Use real transmission patterns")
    elif len(transmission_data) >= 2:
        print("⚠️  Limited: Found some transmission data but not enough for 10 clients")
        print("💡 Recommendation: Use data augmentation or synthetic patterns")
        print("   - Duplicate and time-shift existing patterns")
        print("   - Add randomized availability intervals")
        print("   - Use different CSV files for different clients")
    else:
        print("❌ Insufficient: Very limited transmission data")
        print("💡 Recommendation: Create synthetic availability patterns")
        print("   - Generate random on/off patterns")
        print("   - Use simple periodic availability")
        print("   - Focus on testing the FL integration mechanism")
    
    print("\n💡 Configuration suggestions:")
    print("   - Set min_fit_clients to a lower value (e.g., 2-3)")
    print("   - Use shorter round durations to increase availability chances")
    print("   - Consider different time_scale values")
    print("   - Enable randomize_mapping for domain randomization")

if __name__ == "__main__":
    main()
