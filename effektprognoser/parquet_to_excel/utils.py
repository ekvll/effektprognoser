import pandas as pd


def drop_column(df: pd.DataFrame, drop_col: str = "_left") -> pd.DataFrame:
    for col in df.columns:
        if drop_col in col:
            df.drop(columns=col, inplace=True)
    return df


def merge_dataframes(
    df_left: pd.DataFrame, df_right: pd.DataFrame, suffixes: tuple = ("_left", "")
) -> pd.DataFrame:
    merged_df = pd.merge(
        df_left,
        df_right,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=suffixes,
    )
    dropped_rows = df_right.index.difference(df_left.index)

    if len(dropped_rows) > 0:
        print(f"Dropped rows: {dropped_rows}")

    return merged_df


def get_kommunkod_by_kommunnamn(df, kommunnamn):
    result = df.loc[df["kn"] == kommunnamn, "kk"]
    return result.iloc[0] if not result.empty else None
