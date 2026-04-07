import pandas as pd
import numpy as np

def create_features(data_transformed):
    df = data_transformed[[
        "con",
        "fdpp_partition_date",
        "balance_in_functional_currency",
        "nominal_account_number_mean",
        "nominal_account_number_median"
    ]].copy()

    # Ensure proper datetime (month-end dates)
    df["fdpp_partition_date"] = pd.to_datetime(df["fdpp_partition_date"])

    # Compute the expected previous month-end for each row
    df["prev_month_end"] = df["fdpp_partition_date"] - pd.offsets.MonthEnd(1)

    # Build a "previous month" lookup table
    prev = df.rename(columns={
        "prev_month_end": "mock",
        "fdpp_partition_date": "prev_month_end",
        "balance_in_functional_currency": "bal_prev",
        "nominal_account_number_mean": "mean_prev",
        "nominal_account_number_median": "median_prev"
    })[["con", "mock", "prev_month_end", "bal_prev", "mean_prev", "median_prev"]]

    # Left join: only matches when the exact previous month exists; otherwise NaN (gap handled)
    df = df.merge(prev, on=["con", "prev_month_end"], how="left")

    def safe_div(n, d):
        n = pd.to_numeric(n, errors="coerce")
        d = pd.to_numeric(d, errors="coerce")

        out = np.where((d == 0) | d.isna() | n.isna(), 0.0, n / d)
        out = pd.Series(out, index=n.index)
        return out.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Current-to-previous-month ratios (only populated when previous month exists)
    df["bal_over_bal_prev"]     = safe_div(df["balance_in_functional_currency"], df["bal_prev"])
    df["bal_over_mean_prev"]    = safe_div(df["balance_in_functional_currency"], df["mean_prev"])
    df["bal_over_median_prev"]  = safe_div(df["balance_in_functional_currency"], df["median_prev"])

    # Current-to-current ratios
    df["bal_over_mean_curr"]    = safe_div(df["balance_in_functional_currency"], df["nominal_account_number_mean"])
    df["bal_over_median_curr"]  = safe_div(df["balance_in_functional_currency"], df["nominal_account_number_median"])

    df_features = df[[
        "con", "fdpp_partition_date",
        # "balance_in_functional_currency",
        # "nominal_account_number_mean", "nominal_account_number_median",
        "bal_over_bal_prev",
        "bal_over_mean_prev", "bal_over_median_prev",
        "bal_over_mean_curr", "bal_over_median_curr"
    ]].sort_values(["con", "fdpp_partition_date"])

    return df_features


    import datetime

target_value = "balance_in_functional_currency"

def performe_shrinkage(data: pd.DataFrame):
    
    data = data.astype({target_value: "float64"})
    
    def create_surrounding(data: pd.DataFrame, target_value: str) -> pd.DataFrame:
        group_cols = ["nominal_account_number", "fdpp_partition_date"]
        
        agg = (data.groupby(group_cols)[target_value]
               .agg(median="median", mean="mean") #, count="count"
               .reset_index())
        
        agg = agg.rename(columns={
            "median": "nominal_account_number_median",
            "mean": "nominal_account_number_mean",
            # "count": "nominal_account_number_count",
        })
        
        return data.merge(agg, on=group_cols, how="left")
        
    data = create_surrounding(data, target_value)
    
    return data

data_transformed = performe_shrinkage(data)