import pandas as pd
import prince
from sklearn.pipeline import Pipeline

def extract_categorical_features(
    df: pd.DataFrame, 
    n_components: int = 3
) -> pd.DataFrame:
    """
    Performs Multiple Correspondence Analysis on categorical data.
    Returns a DataFrame with continuous principal components.

    Small n_iter (e.g., 2-4): Computations are lightning-fast. This is fully sufficient when the first few components (e.g., 2 or 3) contain the vast majority of information in your data, and the signal strongly separates from the noise. In the previous code, we used a value of 3 to demonstrate speed.

    Large n_iter (e.g., 7-15): Increases computation time but improves approximation precision. This is necessary when the so-called singular value spectrum drops very slowly - meaning when many dimensions are equally important and the data is highly noisy.

    Modern scikit-learn standards typically set the default value of n_iter=5 (or 7, depending on the chosen solver).
    """
    
    # Initialize MCA model from the prince library.
    # Uses 'sklearn' engine under the hood for optimized SVD.
    mca = prince.MCA(
        n_components=n_components,
        n_iter=3,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    
    # prince works well with modern pandas.
    # The fit_transform method directly returns the projection coordinates.
    mca_coordinates = mca.fit_transform(df)
    
    # Rename columns to be more readable
    mca_coordinates.columns = [f"MCA_component_{i}" for i in range(n_components)]
    
    return mca_coordinates

# Example usage:
df_categorical = pd.DataFrame({
   'color': ['red', 'green', 'red', 'blue'],
   'size': ['S', 'M', 'L', 'S'],
   'country': ['PL', 'DE', 'PL', 'US']
})

continuous_features = extract_categorical_features(df_categorical)


print(df_categorical)
print(continuous_features)