import struct
import os

SHARD_FORMAT = '<QffBBfhB'  # Timestamp (u64), Price (f32), Volatility (f32), Regime (u8), Action (u8), PnL (f32), Momentum_scaled (i16), VolumeDivFlag (u8)
RECORD_SIZE = struct.calcsize(SHARD_FORMAT)  # Should be 26 bytes

def display_shards(file_path: str, num_shards_to_display: int = 10):
    """
    Display the first num_shards_to_display Wisdom Shards from the binary log file.
    Args:
        file_path (str): Path to the binary shard file.
        num_shards_to_display (int): Number of records to display.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    try:
        with open(file_path, 'rb') as f:
            for i in range(num_shards_to_display):
                data = f.read(RECORD_SIZE)
                if len(data) < RECORD_SIZE:
                    print(f"End of file or incomplete record at shard {i+1}.")
                    break
                try:
                    unpacked = struct.unpack(SHARD_FORMAT, data)
                except struct.error as e:
                    print(f"Error unpacking shard {i+1}: {e}")
                    break
                (timestamp_int, price, volatility, regime_prediction, action_taken, pnl_realized, micro_momentum_scaled, volume_divergence_flag) = unpacked
                micro_momentum = micro_momentum_scaled / 10000.0
                print(f"Shard {i+1}: TS={timestamp_int}, Price={price:.4f}, Vol={volatility:.6f}, Regime={regime_prediction}, Action={action_taken}, PnL={pnl_realized:.4f}, Momentum={micro_momentum:.6f}, VolDiv={volume_divergence_flag}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading shards: {e}")

if __name__ == "__main__":
    display_shards("data/shards.bin", 10) 