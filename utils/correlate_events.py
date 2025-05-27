import argparse
import struct
import os
import re
from typing import List, Tuple, Dict, Any
from datetime import datetime

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

# Key log event regexes
LOG_PATTERNS = [
    # (event_type, regex, groupdict mapping)
    ('AnomalyReport', re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*AnomalyReport: type=(?P<atype>[^,]+), strength=(?P<astr>[^,]+), features=(?P<afeat>[^,]+), penalty=(?P<apen>[^\s]+)'), ['ts','atype','astr','afeat','apen']),
    ('L1Trigger', re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*FSM WAKE TRIGGER!'), ['ts']),
    ('ConfidencePenalty', re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Confidence penalty applied: original=(?P<orig>[^,]+), penalty=(?P<pen>[^,]+), adjusted=(?P<adj>[^\s]+)'), ['ts','orig','pen','adj']),
    ('LatencyCircuitBreaker', re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*DEFENSIVE STANCE TRIGGERED: High Latency Volatility.*Using defensive FSM wake threshold:'), ['ts']),
    ('CrossValidation', re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*CROSS-VALIDATION WARNING:.*Reducing confidence from (?P<prev>[^ ]+) to (?P<adj>[^\s]+)'), ['ts','prev','adj']),
    ('MemoryWatchdog', re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*MemoryWatchdog: RSS=(?P<rss>\d+)'), ['ts','rss']),
]

def parse_args():
    parser = argparse.ArgumentParser(description='Unified Event Correlator for Crypto Bot')
    parser.add_argument('--shards', default='data/shards.bin', help='Path to shards.bin file')
    parser.add_argument('--log', default='project_crypto_bot_peter.log', help='Path to log file')
    parser.add_argument('--window', type=int, default=3, help='Correlation window in seconds (+/-)')
    return parser.parse_args()

def decode_shards(file_path: str) -> List[dict]:
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    file_size = os.path.getsize(file_path)
    n_shards = file_size // SHARD_SIZE
    if n_shards == 0:
        print("No shards found or file is empty.")
        return []
    with open(file_path, 'rb') as f:
        raw_data = f.read(n_shards * SHARD_SIZE)
        decoded = []
        for i in range(n_shards):
            try:
                shard = struct.unpack(SHARD_FORMAT, raw_data[i*SHARD_SIZE:(i+1)*SHARD_SIZE])
                record = dict(zip(SHARD_FIELDS, shard))
                decoded.append(record)
            except struct.error as e:
                print(f"Warning: struct.error at shard {i}: {e}")
        return decoded

def parse_log_events(log_path: str) -> List[Dict[str, Any]]:
    events = []
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return events
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            for event_type, regex, fields in LOG_PATTERNS:
                m = regex.match(line)
                if m:
                    event = {'event_type': event_type}
                    for field in fields:
                        event[field] = m.group(field) if field in m.groupdict() else None
                    # Parse timestamp to epoch
                    if 'ts' in event:
                        try:
                            dt = datetime.strptime(event['ts'], '%Y-%m-%d %H:%M:%S,%f')
                            event['timestamp'] = int(dt.timestamp())
                        except Exception:
                            event['timestamp'] = None
                    events.append(event)
                    break
    return sorted([e for e in events if e.get('timestamp') is not None], key=lambda x: x['timestamp'])

def find_nearby_shards(event_ts: int, shards: List[dict], window: int) -> List[dict]:
    return [s for s in shards if abs(int(s['timestamp']) - event_ts) <= window]

def print_event_and_shards(event: Dict[str, Any], shards: List[dict], window: int):
    print(f"\n=== {event['event_type']} @ {event.get('ts','?')} ===")
    for k, v in event.items():
        if k not in ('event_type','ts','timestamp') and v is not None:
            print(f"  {k}: {v}")
    if not shards:
        print(f"  No Wisdom Shards within +/-{window}s.")
    else:
        for s in shards:
            print(f"  Shard ts={s['timestamp']}: price={s['price']:.2f}, vol={s['live_volume']:.2f}, ema={s['ema_price']:.2f}, volat={s['calculated_volatility']:.4f}, vwma_dev={s['vwma_deviation_pct']:.3f}, lvi={s['latency_volatility_index']}")

def main():
    args = parse_args()
    shards = decode_shards(args.shards)
    log_events = parse_log_events(args.log)
    if not log_events:
        print("No significant log events found.")
        return
    for event in log_events:
        nearby_shards = find_nearby_shards(event['timestamp'], shards, args.window)
        print_event_and_shards(event, nearby_shards, args.window)

if __name__ == '__main__':
    main() 