import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import re
import sys

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
