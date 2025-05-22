from tqdm import tqdm
from pathlib import Path

from ep.config import PARQUET_DIR


def main(region: str) -> None:
    tqdm.write(f"Processing region: {region}")
    path: str = Path(PARQUET_DIR) / region


if __name__ == "__main__":
    regions = ["10"]
    for region in regions:
        main(region)
