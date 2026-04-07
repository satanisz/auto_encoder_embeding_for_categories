import pandas as pd
import numpy as np

from agregation import safe_div, equalish_intervals_1_to_n, perform_shrinkage_simple

def example_safe_div() -> None:
    """
    Example demonstrating division handling with Pandas Series.
    It covers normal division, division by zero, null values (NaN), and infinities.
    """
    print("--- safe_div Example ---")
    numerators = pd.Series([10.0, 20.0, 0.0, 5.0, np.nan, 30.0])
    denominators = pd.Series([2.0, 0.0, 5.0, np.nan, 10.0, np.inf])
    
    result_div = safe_div(numerators, denominators)
    
    df_div = pd.DataFrame({
        "numerator": numerators,
        "denominator": denominators,
        "safe_div_result": result_div
    })
    print(df_div)
    print("\n")


def example_equalish_intervals_1_to_n() -> None:
    """
    Example demonstrating equalish intervals calculation.
    
    Creates sample populations (e.g., item counts for different categories) and 
    calculates 4 intervals (will create 5 edges per category: m1, m2, m3, m4, m5).
    """
    print("--- equalish_intervals_1_to_n Example ---")
    populations = pd.Series([10, 25, 2, 100], index=['Category_A', 'Category_B', 'Category_C', 'Category_D'])
    
    intervals_df = equalish_intervals_1_to_n(populations, intervals=4)
    
    print("Populations (Max bounds):")
    print(populations)
    print("\nCalculated Interval Edges:")
    print(intervals_df)


def example_perform_shrinkage() -> None:
    """
    Example demonstrating target encoding with shrinkage.
    Groups by categorical columns and partition date to calculate smoothed target measurements.
    """
    print("\n--- perform_shrinkage Example ---")
    
    # Create sample DataFrame
    data = pd.DataFrame({
        "fdpp_partition_date": ["2026-01-01"] * 10,
        "con": range(10, 20),
        "category_A": ["cat1", "cat1", "cat1", "cat2", "cat2", "cat2", "cat2", "cat3", "cat3", "cat3"],
        "category_B": ["typeX", "typeY", "typeX", "typeY", "typeX", "typeX", "typeY", "typeY", "typeX", "typeX"],
        "balance_in_functional_currency": [100.0, 150.0, 120.0, 50.0, 60.0, 55.0, 45.0, 200.0, 210.0, 190.0]
    })
    
    print("Original Data:")
    print(data)
    
    columns_to_aggregate = ["category_A", "category_B"]
    
    # Perform shrinkage
    result_df = perform_shrinkage_simple(
        data=data, 
        columns_to_aggregate=columns_to_aggregate,
        agg_type_list=["mean", "count", "std"],
        target_value="balance_in_functional_currency",
        intervals=5
    )
    
    print("\nResulting Shrinkage Dataframe (showing subset of columns to avoid clutter):")
    cols_to_show = ["fdpp_partition_date", "con", "category_A_mu_0", "category_A_mu_1", "category_A_mu_2"]
    print(result_df[cols_to_show])

if __name__ == "__main__":
    example_safe_div()
    example_equalish_intervals_1_to_n()
    example_perform_shrinkage()

