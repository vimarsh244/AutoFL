import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import re
import sys
import os

# Increase CSV field size limit to handle long vectime/vecvalue fields
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# A simple availability scheduler built from OMNeT++ CSVs

@dataclass
class Interval:
    start_s: float
    end_s: float

    def to_rounds(self, round_duration_s: float, start_time_s: float = 0.0) -> Tuple[int, int]:
        rs = int(max(1, ((self.start_s - start_time_s) // round_duration_s) + 1))
        re = int(max(1, ((self.end_s - start_time_s) // round_duration_s) + 1))
        return rs, re

class AvailabilityScheduler:
    def __init__(
        self,
        client_intervals_s: Dict[int, List[Interval]],
        round_duration_s: float = 1.0,
        start_time_s: float = 0.0,
        randomize_mapping: bool = False,
        num_clients: Optional[int] = None,
        rnd_seed: int = 42,
    ) -> None:
        self.client_intervals_s = client_intervals_s
        self.round_duration_s = round_duration_s
        self.start_time_s = start_time_s
        self.randomize_mapping = randomize_mapping
        self.num_clients = num_clients or (max(client_intervals_s.keys()) + 1 if client_intervals_s else 0)
        self.rng = random.Random(rnd_seed)

    def is_available(self, client_id: int, round_idx: int) -> bool:
        # If randomize_mapping, remap client_id each round
        mapped_id = self._map_client(client_id, round_idx)
        intervals = self.client_intervals_s.get(mapped_id, [])
        for it in intervals:
            rs, re = it.to_rounds(self.round_duration_s, self.start_time_s)
            if rs <= round_idx <= re:
                return True
        return False

    def get_available_clients(self, round_idx: int) -> List[int]:
        avail = []
        for cid in range(self.num_clients):
            if self.is_available(cid, round_idx):
                avail.append(cid)
        return avail

    def _map_client(self, client_id: int, round_idx: int) -> int:
        if not self.randomize_mapping:
            return client_id
        # Simple deterministic randomization per round to provide domain randomization
        self.rng.seed((round_idx + 1) * 1000003 + client_id)
        return self.rng.randrange(0, self.num_clients)

# ------------------ CSV Parsing ------------------

# Signal name filters to match OMNeT++ vector names
SIGNAL_NAME_FILTERS = {
    "transmissionState": re.compile(r"transmission(State|Stat)", re.IGNORECASE),
    "rx": re.compile(r"(rx(bit|byte|Throughput|Pkt|Goodput)|received)", re.IGNORECASE),
    "tx": re.compile(r"(tx(bit|byte|Throughput|Pkt)|sent)", re.IGNORECASE),
    "sinr": re.compile(r"sinr", re.IGNORECASE),
}

SUPPORTED_SIGNALS = set(SIGNAL_NAME_FILTERS.keys())


def _split_nums(s: str) -> List[float]:
    # Remove surrounding quotes and collapse delimiters
    if s is None:
        return []
    s = s.strip().strip('"').replace("\n", " ")
    for d in [',', ';', '\t']:
        s = s.replace(d, ' ')
    parts = [p for p in s.split() if p]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            # ignore tokens like 'nan' or non-numeric
            try:
                if p.lower() == 'nan':
                    vals.append(float('nan'))
            except Exception:
                pass
    return vals

# Heuristics to map a module/name to a car index (car[<idx>] or containing that)

def _extract_car_index(module: str) -> Optional[int]:
    if not module:
        return None
    # Common patterns: 'car[3]', 'Highway.car[5].*', possibly 'NRCar' with index
    m = re.search(r"car\[(\d+)\]", module)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    
    # For infrastructure like Highway.router.ppp[1].ppp, map ppp[X] to car index
    m = re.search(r"ppp\[(\d+)\]", module)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    
    # For upf or other infrastructure, assign a fixed index
    if "upf" in module.lower():
        return 0  # Map upf to car 0
    
    return None

@dataclass
class ParsedVector:
    module: str
    name: str
    times: List[float]
    values: List[float]


def parse_omnet_csv(file_path: str) -> List[ParsedVector]:
    vectors: Dict[Tuple[str, str], ParsedVector] = {}

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    print(f"[Parser] Reading {path}")
    row_count = 0
    vector_rows = 0
    
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        print(f"[Parser] CSV columns: {reader.fieldnames}")
        
        # Some CSVs may have trailing columns; DictReader handles by fieldnames
        for row in reader:
            row_count += 1
            if row_count <= 3:  # Show first few rows for debugging
                print(f"[Parser] Row {row_count}: {dict(row)}")
                
            row_type = (row.get("type") or "").strip()
            if row_type != "vector":
                continue
            
            vector_rows += 1
            module = (row.get("module") or "").strip()
            name = (row.get("name") or "").strip()
            # Some exports name vectors like 'transmissionState:vector'
            if name.endswith(":vector"):
                name = name[:-8]
            vt = row.get("vectime")
            vv = row.get("vecvalue")
            times = _split_nums(vt) if vt else []
            values = _split_nums(vv) if vv else []
            if not times and not values:
                # Some tools might store a single 'value' per row; treat it as one sample at 0
                single_v = row.get("value")
                if single_v is not None:
                    try:
                        values = [float(single_v)]
                        times = [0.0]
                    except Exception:
                        continue
            if not times or not values:
                continue
            # Trim to shortest length
            L = min(len(times), len(values))
            times = times[:L]
            values = values[:L]
            key = (module, name)
            if key not in vectors:
                vectors[key] = ParsedVector(module, name, [], [])
            vectors[key].times.extend(times)
            vectors[key].values.extend(values)

    print(f"[Parser] Processed {row_count} total rows, {vector_rows} vector rows, {len(vectors)} unique vectors")
    
    # Sort times/values per vector
    result = []
    for pv in vectors.values():
        if pv.times:
            paired = sorted(zip(pv.times, pv.values), key=lambda x: x[0])
            times, values = zip(*paired)
            result.append(ParsedVector(pv.module, pv.name, list(times), list(values)))
    return result


def build_intervals_from_vectors(
    vectors: List[ParsedVector],
    signal: str = "transmissionState",
    threshold: Optional[float] = None,
    min_gap_s: float = 0.0,
    time_scale: float = 1.0,
    max_clients: Optional[int] = None,
) -> Dict[int, List[Interval]]:
    # Validate signal
    if signal not in SUPPORTED_SIGNALS:
        raise ValueError(f"Unsupported signal: {signal}")

    name_filter = SIGNAL_NAME_FILTERS[signal]

    def interpret(v: float) -> bool:
        try:
            fv = float(v)
        except Exception:
            return False
        if signal == 'sinr' and threshold is not None:
            return fv > float(threshold)
        return fv > 0.0

    by_car: Dict[int, List[Tuple[float, bool]]] = {}
    for pv in vectors:
        # Filter by signal name
        if not name_filter.search(pv.name):
            continue
        car_idx = _extract_car_index(pv.module)
        if car_idx is None:
            continue
        if max_clients is not None and car_idx >= max_clients:
            continue
        states: List[Tuple[float, bool]] = []
        for t, val in zip(pv.times, pv.values):
            on = interpret(val)
            states.append((t * time_scale, on))
        if not states:
            continue
        by_car.setdefault(car_idx, [])
        by_car[car_idx].extend(states)

    # Build on-intervals per car from sample-hold states
    intervals_by_car: Dict[int, List[Interval]] = {}
    for car, samples in by_car.items():
        samples.sort(key=lambda x: x[0])
        intervals: List[Interval] = []
        current_start: Optional[float] = None
        current_state = False
        last_t: Optional[float] = None
        for t, on in samples:
            if current_state and not on:
                # closing interval
                if current_start is not None and (t - current_start) >= min_gap_s:
                    intervals.append(Interval(current_start, t))
                current_start = None
            if on and not current_state:
                # opening interval
                current_start = t
            current_state = on
            last_t = t
        # If ended in ON state, close at last_t
        if current_state and current_start is not None and last_t is not None:
            intervals.append(Interval(current_start, last_t))
        intervals_by_car[car] = intervals

    return intervals_by_car


def load_scheduler_from_csvs(
    csv_paths: List[str],
    signal: str = "transmissionState",
    threshold: Optional[float] = None,
    time_scale: float = 1.0,
    round_duration_s: float = 1.0,
    start_time_s: float = 0.0,
    randomize_mapping: bool = False,
    num_clients: Optional[int] = None,
    rnd_seed: int = 42,
    min_gap_s: float = 0.0,
) -> AvailabilityScheduler:
    vectors_all: List[ParsedVector] = []
    for p in csv_paths:
        try:
            parsed = parse_omnet_csv(p)
            print(f"[Scheduler] Parsed {len(parsed)} vectors from {p}")
            vectors_all.extend(parsed)
        except FileNotFoundError:
            print(f"[Scheduler] CSV file not found: {p}")
            continue
        except Exception as e:
            print(f"[Scheduler] Error parsing {p}: {e}")
            continue
    
    print(f"[Scheduler] Total vectors: {len(vectors_all)}")
    print(f"[Scheduler] Looking for signal: {signal}")
    
    # Debug: show first few vectors
    for i, v in enumerate(vectors_all[:5]):
        print(f"[Scheduler] Vector {i}: module='{v.module}', name='{v.name}', {len(v.times)} samples")
    
    intervals_by_car = build_intervals_from_vectors(
        vectors_all,
        signal=signal,
        threshold=threshold,
        time_scale=time_scale,
        max_clients=num_clients,
        min_gap_s=min_gap_s,
    )
    
    print(f"[Scheduler] Found intervals for {len(intervals_by_car)} cars: {list(intervals_by_car.keys())}")
    
    # If no intervals found, create a fallback schedule where all clients are always available
    if not intervals_by_car and num_clients:
        print(f"[Scheduler] No vectors found, creating fallback schedule for {num_clients} clients")
        # Create long availability intervals for all clients
        from dataclasses import dataclass
        fallback_interval = Interval(start_s=0.0, end_s=1000000.0)  # Very long interval
        intervals_by_car = {i: [fallback_interval] for i in range(num_clients)}
    
    return AvailabilityScheduler(
        client_intervals_s=intervals_by_car,
        round_duration_s=round_duration_s,
        start_time_s=start_time_s,
        randomize_mapping=randomize_mapping,
        num_clients=num_clients,
        rnd_seed=rnd_seed,
    )


def parse_omnet_csv(file_path: str) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """Enhanced CSV parser specifically for Car2BS simulation data"""
    import csv
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        print(f"⚠ File not found: {file_path}")
        return {}
    
    car_data = {}
    
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_type = (row.get("type") or "").strip()
            if row_type != "vector":
                continue
                
            module = (row.get("module") or "").strip()
            name = (row.get("name") or "").strip()
            
            # Look for transmission state vectors
            if "transmissionState" not in name:
                continue
                
            # Extract car identifier from module
            car_id = _extract_car_identifier(module)
            if not car_id:
                car_id = "unknown"
            
            # Parse time and value vectors
            vectime = row.get("vectime", "")
            vecvalue = row.get("vecvalue", "")
            
            if vectime and vecvalue:
                times = _split_nums(vectime)
                values = _split_nums(vecvalue)
                
                if times and values and len(times) == len(values):
                    # Convert to availability intervals
                    intervals = _convert_to_intervals(times, values)
                    
                    if intervals:
                        if car_id not in car_data:
                            car_data[car_id] = {'intervals': []}
                        car_data[car_id]['intervals'].extend(intervals)
    
    # Remove duplicates and sort intervals for each car
    for car_id in car_data:
        intervals = car_data[car_id]['intervals']
        # Remove duplicates and sort
        unique_intervals = list(set(intervals))
        unique_intervals.sort(key=lambda x: x[0])
        car_data[car_id]['intervals'] = unique_intervals
    
    return car_data


def _extract_car_identifier(module: str) -> str:
    """Extract car identifier from module name"""
    import re
    
    # Try different patterns to extract car info
    patterns = [
        r'car\[(\d+)\]',           # car[1]
        r'car(\d+)',               # car1  
        r'Highway\.(\w+)',         # Highway.upf, Highway.router
        r'\.(\w+)\.',              # Any module name between dots
    ]
    
    for pattern in patterns:
        match = re.search(pattern, module)
        if match:
            return match.group(1)
    
    # If no pattern matches, use the whole module as identifier
    return module.replace(".", "_").replace("[", "_").replace("]", "")


def _convert_to_intervals(times: List[float], values: List[float]) -> List[Tuple[float, float]]:
    """Convert time series to availability intervals"""
    intervals = []
    current_start = None
    
    for i, (time, value) in enumerate(zip(times, values)):
        is_active = value > 0.5  # Consider > 0.5 as active/available
        
        if is_active and current_start is None:
            # Start of availability interval
            current_start = time
        elif not is_active and current_start is not None:
            # End of availability interval
            intervals.append((current_start, time))
            current_start = None
    
    # If we end in an active state, close the interval
    if current_start is not None and times:
        intervals.append((current_start, times[-1]))
    
    return intervals


def create_enhanced_simulator(config):
    """Create enhanced simulator with multiple data augmentation strategies"""
    import random
    import os
    
    files_data = {}
    total_cars = 0
    
    # Load all available CSV files (prioritize high-quality data first)
    csv_files = [
        "Cars2BS_Sim_Data/Car2BS_Mean_SD.csv",        # High quality - 24,965 points
        "Cars2BS_Sim_Data/Car2BS_Multi_Mean_SD.csv",  # Low quality - 1 point
        "Cars2BS_Sim_Data/BS2Car.csv",                # No transmission vectors
        "Cars2BS_Sim_Data/BS2Car_Mean_SD.csv"         # No transmission vectors
    ]
    
    for csv_path in csv_files:
        full_path = os.path.join(os.path.dirname(__file__), "..", csv_path)
        if os.path.exists(full_path):
            car_data = parse_omnet_csv(full_path)
            if car_data:
                files_data[csv_path] = car_data
                total_cars += len(car_data)
                # Show data quality
                for car_id, data in car_data.items():
                    points = len(data['intervals'])
                    print(f"✓ Loaded car {car_id} from {csv_path}: {points} availability intervals")
            else:
                print(f"⚠ No transmission data found in {csv_path}")
        else:
            print(f"⚠ File not found: {csv_path}")
    
    print(f"📊 Total cars with transmission data: {total_cars}")
    
    if total_cars == 0:
        print("❌ No transmission data found in any CSV file")
        return None
    
    # Create availability scheduler
    scheduler = AvailabilityScheduler(
        client_intervals_s={},
        round_duration_s=config.get('round_duration', 10.0),
        start_time_s=config.get('start_time', 0.0),
        randomize_mapping=config.get('randomize_mapping', True),
        num_clients=config.get('num_clients', 10),
        rnd_seed=config.get('rnd_seed', 42)
    )
    
    # Add data with smart augmentation
    num_clients = config.get('num_clients', 10)
    
    if total_cars < num_clients:
        print(f"🔄 Augmenting {total_cars} cars to create {num_clients} FL clients")
        
        # Collect all available car data, prioritizing high-quality data
        all_car_data = []
        for csv_path in csv_files:
            if csv_path in files_data:
                file_cars = files_data[csv_path]
                all_car_data.extend(file_cars.values())
        
        # Sort by data quality (number of intervals)
        all_car_data.sort(key=lambda x: len(x['intervals']), reverse=True)
        
        # Create interval data for scheduler
        client_intervals = {}
        
        # Apply various augmentation strategies
        for client_id in range(num_clients):
            if client_id < len(all_car_data):
                # Use original data for first clients
                car_data = all_car_data[client_id]
                intervals = [Interval(start, end) for start, end in car_data['intervals']]
                client_intervals[client_id] = intervals
                print(f"🚗 Client {client_id}: Using original data ({len(intervals)} intervals)")
            else:
                # Augment data for additional clients using best available car
                base_car = all_car_data[0]  # Always use highest quality data
                
                # Apply different augmentation strategies based on client ID
                augmentation_type = client_id % 6
                
                if augmentation_type == 0:  # Time shift forward
                    time_shift = random.uniform(200, 800)
                    shifted_intervals = [Interval(start + time_shift, end + time_shift) 
                                       for start, end in base_car['intervals']]
                    client_intervals[client_id] = shifted_intervals
                    print(f"🚗 Client {client_id}: Time-shifted +{time_shift:.1f}s")
                    
                elif augmentation_type == 1:  # Time shift backward
                    time_shift = random.uniform(200, 800)
                    shifted_intervals = [Interval(max(0, start - time_shift), max(0, end - time_shift)) 
                                       for start, end in base_car['intervals']]
                    # Filter out invalid intervals
                    shifted_intervals = [interval for interval in shifted_intervals if interval.end_s > interval.start_s]
                    client_intervals[client_id] = shifted_intervals
                    print(f"🚗 Client {client_id}: Time-shifted -{time_shift:.1f}s")
                    
                elif augmentation_type == 2:  # Time scale compression
                    scale_factor = random.uniform(0.6, 0.9)
                    scaled_intervals = [Interval(start * scale_factor, end * scale_factor) 
                                      for start, end in base_car['intervals']]
                    client_intervals[client_id] = scaled_intervals
                    print(f"🚗 Client {client_id}: Time-scaled {scale_factor:.2f}x")
                    
                elif augmentation_type == 3:  # Time scale expansion
                    scale_factor = random.uniform(1.1, 1.4)
                    scaled_intervals = [Interval(start * scale_factor, end * scale_factor) 
                                      for start, end in base_car['intervals']]
                    client_intervals[client_id] = scaled_intervals
                    print(f"🚗 Client {client_id}: Time-scaled {scale_factor:.2f}x")
                    
                elif augmentation_type == 4:  # Fragment long intervals
                    fragmented = []
                    for start, end in base_car['intervals']:
                        duration = end - start
                        if duration > 100:  # Fragment intervals longer than 100s
                            num_fragments = random.randint(2, 4)
                            fragment_size = duration / num_fragments
                            gap_size = fragment_size * 0.1  # 10% gap between fragments
                            
                            for i in range(num_fragments):
                                frag_start = start + i * fragment_size + i * gap_size
                                frag_end = frag_start + fragment_size - gap_size
                                if frag_end > frag_start:
                                    fragmented.append(Interval(frag_start, frag_end))
                        else:
                            fragmented.append(Interval(start, end))
                    client_intervals[client_id] = fragmented
                    print(f"🚗 Client {client_id}: Fragmented intervals")
                    
                else:  # Add random gaps and noise
                    noisy = []
                    for start, end in base_car['intervals']:
                        duration = end - start
                        # Add small random offsets
                        start_offset = random.uniform(-5, 5)
                        end_offset = random.uniform(-5, 5) 
                        new_start = max(0, start + start_offset)
                        new_end = max(new_start + 1, end + end_offset)
                        
                        # Randomly skip some intervals (create gaps)
                        if random.random() > 0.1:  # Keep 90% of intervals
                            noisy.append(Interval(new_start, new_end))
                    client_intervals[client_id] = noisy
                    print(f"🚗 Client {client_id}: Added noise and gaps")
        
        # Update scheduler with augmented data
        scheduler.client_intervals_s = client_intervals
        
    else:
        # Use original data without augmentation
        client_intervals = {}
        client_id = 0
        for file_cars in files_data.values():
            for car_id, car_data in file_cars.items():
                if client_id < num_clients:
                    intervals = [Interval(start, end) for start, end in car_data['intervals']]
                    client_intervals[client_id] = intervals
                    client_id += 1
        scheduler.client_intervals_s = client_intervals
    
    print(f"✅ Created availability simulator with {len(scheduler.client_intervals_s)} clients")
    
    # Show availability preview
    print("📊 Client availability preview:")
    for i in range(min(3, len(scheduler.client_intervals_s))):
        intervals = len(scheduler.client_intervals_s[i])
        total_time = sum(interval.end_s - interval.start_s for interval in scheduler.client_intervals_s[i])
        print(f"   Client {i}: {intervals} intervals, {total_time:.1f}s total available time")
    
    return scheduler
