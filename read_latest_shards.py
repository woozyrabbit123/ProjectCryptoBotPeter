import polars as pl
import struct
import argparse
import os
import sys
from typing import List, Tuple, Any

# Wisdom Shard schema: (field_name, struct_format, size_in_bytes)
SHARD_SCHEMA: List[Tuple[str, str, int]] = [
    ('timestamp', 'Q', 8),
    ('price', 'f', 4),
    ('ema_price', 'f', 4),
    ('calculated_volatility', 'f', 4),
    ('volatility_regime', 'B', 1),
    ('action_taken', 'B', 1),
    ('pnl_realized', 'f', 4),
    ('scaled_micro_momentum', 'h', 2),
    ('volume_divergence_flag', 'B', 1),
    ('live_volume', 'f', 4),
    ('vwma_price', 'f', 4),
    ('vwma_deviation_pct', 'f', 4),
    ('fetch_latency_ms', 'H', 2),
    ('latency_spike_flag', 'B', 1),
    ('latency_volatility_index', 'h', 2),
]

SHARD_FORMAT = '<' + ''.join(fmt for _, fmt, _ in SHARD_SCHEMA)
SHARD_SIZE = struct.calcsize(SHARD_FORMAT)
SHARD_FIELDS = [name for name, _, _ in SHARD_SCHEMA]

LOG_FILE = 'project_crypto_bot_peter.log'


def parse_args():
    parser = argparse.ArgumentParser(description='Wisdom Shard Analysis Utility')
    parser.add_argument('--file', default='data/shards.bin', help='Path to shards.bin file')
    parser.add_argument('--last', type=int, default=20, help='Number of most recent shards to decode')
    parser.add_argument('--from-ts', type=int, help='Start timestamp (inclusive)')
    parser.add_argument('--to-ts', type=int, help='End timestamp (inclusive)')
    parser.add_argument('--csv', action='store_true', help='Output as CSV instead of table')
    return parser.parse_args()


def load_log_events(log_path: str):
    """Parse log file for AnomalyReport and L1 trigger events, return list of (timestamp, event_type)"""
    events = []
    if not os.path.exists(log_path):
        return events
    import re
    ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
    anomaly_re = re.compile(r'AnomalyReport:')
    l1_re = re.compile(r'FSM WAKE TRIGGER!')
    from datetime import datetime
    def parse_ts(line):
        m = ts_re.match(line)
        if m:
            try:
                dt = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                return int(dt.timestamp())
            except Exception:
                return None
        return None
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            ts = parse_ts(line)
            if ts is not None:
                if anomaly_re.search(line):
                    events.append((ts, 'AnomalyNearby'))
                elif l1_re.search(line):
                    events.append((ts, 'L1TriggerNearby'))
    return events


def find_nearby_events(shard_ts: int, events: List[Tuple[int, str]], window: int = 2) -> List[str]:
    """Return list of event types within +/- window seconds of shard_ts"""
    return [etype for ts, etype in events if abs(ts - shard_ts) <= window]


def decode_shards(file_path: str, last_n: int = 20, from_ts: int = None, to_ts: int = None, log_events=None) -> List[dict]:
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    file_size = os.path.getsize(file_path)
    n_shards = file_size // SHARD_SIZE
    if n_shards == 0:
        print("No shards found or file is empty.")
        return []
    # Determine which shards to read
    with open(file_path, 'rb') as f:
        # Read all for range, or just last N
        if from_ts or to_ts:
            # Read all, filter by timestamp
            raw_data = f.read(n_shards * SHARD_SIZE)
            decoded = []
            for i in range(n_shards):
                try:
                    shard = struct.unpack(SHARD_FORMAT, raw_data[i*SHARD_SIZE:(i+1)*SHARD_SIZE])
                    record = dict(zip(SHARD_FIELDS, shard))
                    decoded.append(record)
                except struct.error as e:
                    print(f"Warning: struct.error at shard {i}: {e}")
            # Filter by timestamp
            if from_ts:
                decoded = [r for r in decoded if r['timestamp'] >= from_ts]
            if to_ts:
                decoded = [r for r in decoded if r['timestamp'] <= to_ts]
            return decoded
        else:
            # Only last N
            records_to_read = min(last_n, n_shards)
            if records_to_read == 0:
                return []
            f.seek(-records_to_read * SHARD_SIZE, 2)
            raw_data = f.read(records_to_read * SHARD_SIZE)
            decoded = []
            for i in range(records_to_read):
                try:
                    shard = struct.unpack(SHARD_FORMAT, raw_data[i*SHARD_SIZE:(i+1)*SHARD_SIZE])
                    record = dict(zip(SHARD_FIELDS, shard))
                    decoded.append(record)
                except struct.error as e:
                    print(f"Warning: struct.error at last-{records_to_read-i}: {e}")
            return decoded


def temporal_sanity_check(shards: List[dict]):
    prev_ts = None
    for i, shard in enumerate(shards):
        ts = shard['timestamp']
        if prev_ts is not None:
            if ts < prev_ts:
                print(f"Warning: Timestamp decreased at shard {i}: {ts} < {prev_ts}")
            elif ts - prev_ts > 3600 * 24:
                print(f"Warning: Large timestamp jump at shard {i}: {ts} vs {prev_ts}")
        prev_ts = ts


def main():
    args = parse_args()
    log_events = load_log_events(LOG_FILE)
    shards = decode_shards(args.file, args.last, args.from_ts, args.to_ts, log_events)
    if not shards:
        return
    temporal_sanity_check(shards)
    # Plugin/hook placeholder
    # TODO: Implement plugin hook for shard-level analysis here
    # Cross-stream correlation annotation
    for shard in shards:
        if log_events:
            nearby = find_nearby_events(int(shard['timestamp']), log_events)
            if nearby:
                shard['correlation_note'] = ','.join(nearby)
    df = pl.DataFrame(shards)
    if args.csv:
        print(df.write_csv())
    else:
        print(df)

if __name__ == '__main__':
    main() 