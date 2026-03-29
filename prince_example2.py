import pandas as pd
import prince

def apply_famd(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Performs dimensionality reduction for mixed data (categorical and numerical).
    """
    famd = prince.FAMD(
        n_components=n_components,
        n_iter=3,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    
    # The model automatically recognizes which columns are numerical and which are categorical
    famd.fit(df)
    
    # Projection to the new space (returns embeddings for rows)
    row_embeddings = famd.transform(df)
    
    return row_embeddings

# Example mixed data
data: dict[str, list] = {
    'Status': ['Active', 'Blocked', 'Active', 'Inactive', 'Active'],
    'Account_type': ['Premium', 'Basic', 'Premium', 'Basic', 'Basic'],
    'Session_time_min': [120.5, 5.2, 340.0, 0.0, 45.5]  # Our key numerical value
}

df_mixed = pd.DataFrame(data)

# Apply model
mixed_coords = apply_famd(df_mixed)
print(mixed_coords)