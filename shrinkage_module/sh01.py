import pandas as pd
import numpy as np

def create_features(data):
    df = data[[
        "con",
        "fdpp_partition_date",
        "balance in functional currency",
        "nominal_account_number_sum"
    ]].copy()

    df["fdpp_partition_date"] = pd.to_datetime(df["fdpp_partition_date"])
    df["prev_month_end"] = df["fdpp_partition_date"] - pd.offsets.MonthEnd(1)
    
    prev = df.rename(columns={
        "prev_month_end": "mock",
        "fdpp_partition_date": "prev_month_end",
        "balance in functional currency": "bal_prev",
        "nominal_account_number_sum": "sum_prev"
    })[[
        "con",
        "mock",
        "prev_month_end",
        "bal_prev",
        "sum_prev"
    ]]
    
    df = df.merge(prev, on=["con", "prev_month_end"], how="left")

    def safe_div(n, d):
        n = pd.to_numeric(n, errors="coerce")
        d = pd.to_numeric(d, errors="coerce")

        out = np.where((d == 0) | d.isna() | n.isna(), 0.0, n / d)
        out = pd.Series(out, index=n.index)
        return out.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    df["monthy_acct_growth"] = safe_div(df["nominal_account_number_sum"], df["sum_prev"])

    df_features = df[[
        "con",
        "fdpp_partition_date",
        "monthy_acct_growth",
    ]].sort_values(["con", "fdpp_partition_date", "monthy_acct_growth"])

    return df_features


