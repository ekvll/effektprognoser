import pandas as pd


def calc_lastprofil_effektbehov_elanvandning(df: pd.DataFrame):
    df.drop(columns=["Tidpunkt"], inplace=True)
    transposed = df.groupby("rid")["Elanvandning"].apply(list).reset_index(name="lp")

    transposed["eb"] = transposed["lp"].apply(lambda x: max(x))
    transposed["ea"] = transposed["lp"].apply(lambda x: sum(x))

    return transposed
