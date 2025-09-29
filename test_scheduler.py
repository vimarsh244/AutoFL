#!/usr/bin/env python3
"""
Quick test to debug the OMNeT++ CSV parsing and availability scheduler
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from sim.availability import load_scheduler_from_csvs

def test_scheduler():
    print("Testing CSV parsing and scheduler...")
    
    csv_files = [
        "Cars2BS_Sim_Data/Car2BS_Mean_SD.csv",
        "Cars2BS_Sim_Data/Car2BS_Multi_Mean_SD.csv", 
        "Cars2BS_Sim_Data/BS2Car.csv"
    ]
    
    try:
        scheduler = load_scheduler_from_csvs(
            csv_paths=csv_files,
            signal="transmissionState",
            time_scale=20.0,
            round_duration_s=1.0,
            randomize_mapping=True,
            num_clients=5,
            rnd_seed=42
        )
        
        print(f"\nScheduler created successfully!")
        print(f"Number of clients configured: {scheduler.num_clients}")
        print(f"Client intervals: {len(scheduler.client_intervals_s)} cars")
        
        # Test availability for first few rounds
        for round_idx in range(1, 6):
            available = scheduler.get_available_clients(round_idx)
            print(f"Round {round_idx}: {len(available)} clients available - {available}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scheduler()
