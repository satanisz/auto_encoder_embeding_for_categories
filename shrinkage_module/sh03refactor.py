import pandas as pd
import numpy as np

def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    """Safely divides two Pandas Series, handling zeros, NaNs, and infinities natively."""
    n_num = pd.to_numeric(n, errors="coerce")
    d_num = pd.to_numeric(d, errors="coerce")
    
    # Vectorized division utilizing Pandas index alignment
    out = n_num.div(d_num).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # Mask invalid denominators/numerators directly
    mask = (d_num == 0) | d_num.isna() | n_num.isna()
    return out.mask(mask, 0.0)

def create_surrounding_stats(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Aggregates mean and median for the target value using modern named aggregation."""
    group_cols = ["nominal_account_number", "fdpp_partition_date"]
    
    # Named aggregation handles the rename and avoids the need to reset_index
    agg = data.groupby(group_cols, as_index=False).agg(
        nominal_account_number_median=(target_col, "median"),
        nominal_account_number_mean=(target_col, "mean")
    )
    return data.merge(agg, on=group_cols, how="left")

def perform_shrinkage(data: pd.DataFrame, target_col: str = "balance_in_functional_currency") -> pd.DataFrame:
    """Entry point for calculating surrounding statistics."""
    df = data.astype({target_col: "float64"})
    return create_surrounding_stats(df, target_col)

def create_features(data_transformed: pd.DataFrame) -> pd.DataFrame:
    """Generates period-over-period and cross-sectional ratio features."""
    cols_to_keep = [
        "con",
        "fdpp_partition_date",
        "balance_in_functional_currency",
        "nominal_account_number_mean",
        "nominal_account_number_median"
    ]
    df = data_transformed[cols_to_keep].copy()
    
    df["fdpp_partition_date"] = pd.to_datetime(df["fdpp_partition_date"])
    df["prev_month_end"] = df["fdpp_partition_date"] - pd.offsets.MonthEnd(1)
    
    # Build previous month lookup table
    prev = df.rename(columns={
        "prev_month_end": "mock",
        "fdpp_partition_date": "prev_month_end",
        "balance_in_functional_currency": "bal_prev",
        "nominal_account_number_mean": "mean_prev",
        "nominal_account_number_median": "median_prev"
    })[["con", "mock", "prev_month_end", "bal_prev", "mean_prev", "median_prev"]]
    
    df = df.merge(prev, on=["con", "prev_month_end"], how="left")
    
    # Extract target column for cleaner calculations below
    bal = df["balance_in_functional_currency"]
    
    # Current-to-previous-month ratios
    df["bal_over_bal_prev"]    = safe_div(bal, df["bal_prev"])
    df["bal_over_mean_prev"]   = safe_div(bal, df["mean_prev"])
    df["bal_over_median_prev"] = safe_div(bal, df["median_prev"])
    
    # Current-to-current ratios
    df["bal_over_mean_curr"]   = safe_div(bal, df["nominal_account_number_mean"])
    df["bal_over_median_curr"] = safe_div(bal, df["nominal_account_number_median"])
    
    feature_cols = [
        "con", "fdpp_partition_date",
        "bal_over_bal_prev", "bal_over_mean_prev", "bal_over_median_prev",
        "bal_over_mean_curr", "bal_over_median_curr"
    ]
    
    return df[feature_cols].sort_values(["con", "fdpp_partition_date"])

# Example Execution Workflow:
# data_transformed = perform_shrinkage(data)
# df_features = create_features(data_transformed)