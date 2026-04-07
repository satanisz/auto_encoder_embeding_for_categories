
target_value = "balance in functional currency"

def performe_agregation(data: pd.DataFrame):
    
    data = data.astype({target_value: "float64"})
    
    def create_surrounding(data: pd.DataFrame, target_value: str) -> pd.DataFrame:
        group_cols = ["nominal_account_number", "fdpp_partition_date"]
        
        agg = (data.groupby(group_cols)[target_value]
               .agg(
                   sum="sum"
               )
               .reset_index())
        
        agg = agg.rename(columns={
            "sum": "nominal_account_number_sum",
        })
        
        return data.merge(agg, on=group_cols, how="left")
        
    data = create_surrounding(data, target_value)
    
    return data

ftp_sdi_BS_combined = performe_agregation(ftp_sdi_BS_combined)


target_value = "balance_in_functional_currency"
def performe_shrinkage(data: pd.DataFrame):
    def equalish_intervals_1_to_n(populations, intervals=5, col_prefix="m"):
        """
        populations: pd.Series / 1D array-like
        returns: DataFrame with columns m1..m{intervals+1}
        """
        s = pd.Series(populations)
        n = s.fillna(1).astype(float).to_numpy()
        n = np.floor(n).astype(int)
        n = np.maximum(n, 1)
        k = intervals + 1  # number of edges
        i = np.arange(k)   # 0..intervals
        edges_f = 1 + np.outer((n - 1) / intervals, i)   # shape (len(n), k)
        edges = np.rint(edges_f).astype(int)
        edges[:, 0] = 1
        edges[:, -1] = n
        edges = np.maximum.accumulate(edges, axis=1)
        cols = [f"{col_prefix}{j}" for j in range(1, k + 1)]
        return pd.DataFrame(edges, index=s.index, columns=cols)

    data = data.astype({target_value: "float64"})

    def create_surrounding(data: pd.DataFrame, columns_to_agregate: list[str], target_value: str) -> pd.DataFrame:
        for col in columns_to_agregate:
            group_cols = [col, "fdpp_partition_date"]
            agg = (data.groupby(group_cols)[target_value]
                   .agg(mean="mean", count="count", std="std") #, count="count"
                   .reset_index())
            agg = agg.rename(columns={
                "mean": f"{col}_mean",
                "count": f"{col}_count",
                "std": f"{col}_std",
            })
            data = data.merge(agg, on=group_cols, how="left")
        return data

    data = create_surrounding(data, columns_to_agregate, target_value)

    for col in columns_to_agregate:
        data[f"{col}_mu_0"] = ((data[target_value]) / data[f"{col}_std"]).replace([np.inf, -np.inf], 0.0).fillna(0.0).abs()
        df_m = equalish_intervals_1_to_n(
            data[f"{col}_count"],
            intervals=5,
            col_prefix=f"{col}_m"
        )
        columns_names_m = list(df_m.columns)  # ensures no (6 vs 7) mismatch
        data[columns_names_m] = df_m
        # --- inputs ---
        m      = data[columns_names_m]        # (n_rows, n_m)
        count  = data[f"{col}_count"]         # (n_rows,)
        mean   = data[f"{col}_mean"]          # (n_rows,)
        y      = data[target_value]           # (n_rows,)
        den = m.add(count, axis=0)            # broadcast count across m columns
        w_df = den.rdiv(count, axis=0)        # count / den
        w_df.columns = [f"{col}_w{i}" for i in range(1, w_df.shape[1]+1)]
        mu_df = (w_df.mul(y, axis=0).add((1 - w_df).mul(mean, axis=0))).abs()
        mu_df.columns = [f"{col}_mu_{i}" for i in range(1, mu_df.shape[1]+1)]
        data = pd.concat([data, w_df, mu_df], axis=1)

    columns_to_process_souranding = ["fdpp_partition_date", "con"]+[
        feat
        for col in columns_to_agregate
            for feat in [f"{col}_mu_{idx}" for idx in range(0, 7)]
    ]
    return data[columns_to_process_souranding].copy()