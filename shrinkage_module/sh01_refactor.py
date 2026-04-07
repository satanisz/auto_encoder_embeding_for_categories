import pandas as pd
import numpy as np



def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    """Safely divides two Pandas Series, handling zeros, NaNs, and infinities."""
    n_num = pd.to_numeric(n, errors="coerce")
    d_num = pd.to_numeric(d, errors="coerce")

    # Utilize pandas native division for automatic index alignment
    out = n_num.div(d_num).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Mask division by zero or where either original series was NaN
    mask = (d_num == 0) | d_num.isna() | n_num.isna()
    return out.mask(mask, 0.0)

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    cols_to_keep = [
        "con",
        "fdpp_partition_date",
        "balance in functional currency",
        "nominal_account_number_sum"
    ]
    df = data[cols_to_keep].copy()

    df["fdpp_partition_date"] = pd.to_datetime(df["fdpp_partition_date"])
    df["prev_month_end"] = df["fdpp_partition_date"] - pd.offsets.MonthEnd(1)

    prev = df.rename(columns={
        "prev_month_end": "mock",
        "fdpp_partition_date": "prev_month_end",
        "balance in functional currency": "bal_prev",
        "nominal_account_number_sum": "sum_prev"
    })[["con", "mock", "prev_month_end", "bal_prev", "sum_prev"]]

    df = df.merge(prev, on=["con", "prev_month_end"], how="left")

    df["monthly_acct_growth"] = safe_div(df["nominal_account_number_sum"], df["sum_prev"])

    df_features = df[[
        "con",
        "fdpp_partition_date",
        "monthly_acct_growth",
    ]].sort_values(["con", "fdpp_partition_date", "monthly_acct_growth"])

    return df_features