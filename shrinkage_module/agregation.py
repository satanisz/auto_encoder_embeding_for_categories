import pandas as pd
import numpy as np


def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    """Safely divides two Pandas Series, handling zeros, NaNs, and infinities."""
    n_num = pd.to_numeric(n, errors="coerce")
    d_num = pd.to_numeric(d, errors="coerce")

    out = n_num.div(d_num).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    mask = (d_num == 0) | d_num.isna() | n_num.isna()
    return out.mask(mask, 0.0)

def equalish_intervals_1_to_n(populations: pd.Series, intervals: int = 5, col_prefix: str = "m") -> pd.DataFrame:
    """Calculates equalish intervals from 1 to n."""
    n = populations.fillna(1).astype(float).to_numpy()
    n = np.maximum(np.floor(n).astype(int), 1)
    
    k = intervals + 1
    i = np.arange(k)
    
    edges_f = 1 + np.outer((n - 1) / intervals, i)
    edges = np.rint(edges_f).astype(int)
    edges[:, 0] = 1
    edges[:, -1] = n
    edges = np.maximum.accumulate(edges, axis=1)
    
    cols = [f"{col_prefix}{j}" for j in range(1, k + 1)]
    return pd.DataFrame(edges, index=populations.index, columns=cols)

def create_surrounding_agg(data: pd.DataFrame, group_cols: list[str], aggregations: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Computes specified aggregations grouped by specified columns and merges them back to the original DataFrame."""
    agg = data.groupby(group_cols, as_index=False).agg(**aggregations)
    return data.merge(agg, on=group_cols, how="left")

def perform_aggregation(data: pd.DataFrame) -> pd.DataFrame:
    """Applies predefined sum aggregations over the dataset, casting the target metric to float."""
    target_value = "balance in functional currency"    
    group_cols = ["nominal_account_number", "fdpp_partition_date"]
    data = data.astype({target_value: "float64"})

    aggregations = {"nominal_account_number_sum": (target_value, "sum")}
    return create_surrounding_agg(data, group_cols, aggregations)


def perform_shrinkage(
    data: pd.DataFrame, 
    columns_to_aggregate: list[str], 
    agg_type_list: list[str] = ["mean", "count", "std"],
    target_value: str = "balance_in_functional_currency",
    intervals: int = 5
) -> pd.DataFrame:
    """Calculates smoothed (shrunk) target measurements per designated columns using a population-weighted average."""
    
    df = data.astype({target_value: "float64"}).copy()

    # 1. Create Surrounding
    for col in columns_to_aggregate:
        group_cols = [col, "fdpp_partition_date"]
        aggregations = {f"{col}_{k}": (target_value, k) for k in agg_type_list}
        df = create_surrounding_agg(df, group_cols, aggregations)

    # 2. Process Shrinkage
    for col in columns_to_aggregate:
        std_col, count_col, mean_col = f"{col}_std", f"{col}_count", f"{col}_mean"
        
        df[f"{col}_mu_0"] = (df[target_value] / df[std_col]).replace([np.inf, -np.inf], 0.0).fillna(0.0).abs()
        
        df_m = equalish_intervals_1_to_n(df[count_col], intervals=intervals, col_prefix=f"{col}_m")
        
        m = df_m
        count = df[count_col]
        mean = df[mean_col]
        y = df[target_value]
        
        den = m.add(count, axis=0)
        w_df = den.rdiv(count, axis=0)
        w_df.columns = [f"{col}_w{i}" for i in range(1, len(w_df.columns) + 1)]
        
        mu_df = w_df.mul(y, axis=0).add((1 - w_df).mul(mean, axis=0)).abs()
        mu_df.columns = [f"{col}_mu_{i}" for i in range(1, len(mu_df.columns) + 1)]
        
        df = pd.concat([df, df_m, w_df, mu_df], axis=1)

    # 3. Compile Final Columns
    columns_to_process = ["fdpp_partition_date", "con"] + [
        f"{col}_mu_{idx}" 
        for col in columns_to_aggregate 
        for idx in range(7)
    ]
    
    return df[columns_to_process].copy()