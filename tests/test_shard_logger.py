import pytest
import struct
import os
from src.shard_logger import log_raw_shard

FMT = '<fIBBf'
RECORD_SIZE = struct.calcsize(FMT)


def test_log_raw_shard_single(tmp_path):
    file_path = tmp_path / "test_shards.bin"
    log_raw_shard(
        str(file_path),
        1678886400,
        123.45,
        123.0,
        0.01,
        1,
        2,
        0.0,
        0.0
    )
    # Read and verify
    with open(file_path, 'rb') as f:
        data = f.read()
    assert len(data) > 0


def test_log_raw_shard_multiple(tmp_path):
    file_path = tmp_path / "test_shards_multi.bin"
    entries = [
        (1600000000, 100.0, 100.0, 0.01, 0, 0, 0, 0.0, 0.0),
        (1600000001, 101.5, 101.0, 0.02, 1, 2, 2, 1.23, 0.1),
        (1600000002, 99.9, 99.0, 0.03, 2, 1, 1, -0.5, -0.2),
    ]
    for entry in entries:
        log_raw_shard(str(file_path), *entry)
    with open(file_path, 'rb') as f:
        data = f.read()
    assert len(data) > 0


def test_log_raw_shard_ioerror(monkeypatch, tmp_path):
    # Simulate IOError by making open raise
    def bad_open(*args, **kwargs):
        raise IOError("Simulated file error")
    monkeypatch.setattr("builtins.open", bad_open)
    file_path = tmp_path / "should_not_exist.bin"
    # Should not raise, just log error
    log_raw_shard(
        str(file_path),
        1,
        1.0,
        1.0,
        0.01,
        0,
        1,
        0.0,
        0.0
    ) 