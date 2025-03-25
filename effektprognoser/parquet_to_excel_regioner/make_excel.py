import os
import pandas as pd
from effektprognoser.paths import DATA_DIR


def main(region):
    input_path = os.path.join(DATA_DIR, "tmp")
    files = os.listdir(input_path)
    print(files)
    file = [f for f in files if region in f][0]

    df = pd.read_parquet(os.path.join(input_path, file))
    print(df.head())


if __name__ == "__main__":
    region = "10"
    main(region)
