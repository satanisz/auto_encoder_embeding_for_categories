import pandas as pd

def create_surrounding_agg(data: pd.DataFrame, target_value: str) -> pd.DataFrame:
    group_cols = ["nominal_account_number", "fdpp_partition_date"]
    
    # Modern named aggregation handles renaming and resetting index in one step
    agg = data.groupby(group_cols, as_index=False).agg(
        nominal_account_number_sum=(target_value, "sum")
    )
    return data.merge(agg, on=group_cols, how="left")

def perform_aggregation(data: pd.DataFrame, target_value: str = "balance in functional currency") -> pd.DataFrame:
    df = data.astype({target_value: "float64"})
    return create_surrounding_agg(df, target_value)

# Execution
# ftp_sdi_BS_combined = perform_aggregation(ftp_sdi_BS_combined)


import pandas as pd
import numpy as np

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

def perform_shrinkage(
    data: pd.DataFrame, 
    columns_to_aggregate: list[str], 
    target_value: str = "balance_in_functional_currency"
) -> pd.DataFrame:
    
    df = data.astype({target_value: "float64"}).copy()

    # 1. Create Surrounding
    for col in columns_to_aggregate:
        group_cols = [col, "fdpp_partition_date"]
        # Leveraging named aggregations with dictionary unpacking
        agg = df.groupby(group_cols, as_index=False).agg(**{
            f"{col}_mean": (target_value, "mean"),
            f"{col}_count": (target_value, "count"),
            f"{col}_std": (target_value, "std")
        })
        df = df.merge(agg, on=group_cols, how="left")

    # 2. Process Shrinkage
    for col in columns_to_aggregate:
        std_col, count_col, mean_col = f"{col}_std", f"{col}_count", f"{col}_mean"
        
        df[f"{col}_mu_0"] = (df[target_value] / df[std_col]).replace([np.inf, -np.inf], 0.0).fillna(0.0).abs()
        
        df_m = equalish_intervals_1_to_n(df[count_col], intervals=5, col_prefix=f"{col}_m")
        
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