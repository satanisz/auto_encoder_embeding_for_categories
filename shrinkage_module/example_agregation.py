import pandas as pd
import numpy as np

from agregation import safe_div, equalish_intervals_1_to_n

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


if __name__ == "__main__":
    example_safe_div()
    example_equalish_intervals_1_to_n()

