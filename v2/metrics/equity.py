import pandas as pd


def load_equity_curve(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["equity"] = df["balance"]
    return df
