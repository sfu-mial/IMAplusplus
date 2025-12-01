import hashlib
from pathlib import Path

import pandas as pd


def calculate_md5_file(path: Path, chunk_size: int = 8192) -> str:
    """
    Calculate the MD5 hash of a file.
    """
    md5 = hashlib.md5()
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def calculate_shortest_md5_length(df: pd.DataFrame, column: str) -> int:
    """
    Calculate the shortest possible truncation of the MD5 hash of a column in
    a DataFrame to ensure that the truncated MD5 hash is unique.
    """
    hashes = df[column].astype(str)

    # Start with the shortest possible truncation length, and increment if
    # there are collisions.
    print("Calculating the shortest possible truncation of the MD5 hash.")
    n = 1
    while n <= 32:  # Since the MD5 hash is 32 characters long.
        print(f"Trying truncation length {n}.")
        if hashes.str[:n].nunique() == len(hashes):
            print(f"Shortest possible truncation of the MD5 hash found: {n}")
            return n
        n += 1
