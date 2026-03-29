import pandas as pd
from sklearn.ensemble import IsolationForest
import prince

def optimize_n_components_by_stability(
    df: pd.DataFrame, 
    components_range: range, 
    top_k_percent: float = 0.05
) -> pd.DataFrame:
    """
    Optimizes the number of components by checking the stability of the top K% anomaly ranking.

    By analyzing the resulting stability_df dataframe, you look for a point where the IoU_vs_previous metric achieves high and stable values (e.g. > 0.85). This means that adding more components no longer adds new knowledge to the anomaly detector.
    """
    k_threshold = max(1, int(len(df) * top_k_percent))
    
    # Store sets of indices (row IDs) identified as top anomalies
    anomaly_sets: dict[int, set[Any]] = {}
    stability_scores: list[dict[str, float]] = []

    for n in components_range:
        # 1. Space transformation
        mca = prince.MCA(n_components=n, random_state=42, engine='sklearn')
        embeddings = mca.fit_transform(df)
        
        # 2. Detection
        iso_forest = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
        iso_forest.fit(embeddings)
        
        # decision_function: lower score = stronger anomaly
        scores = iso_forest.decision_function(embeddings)
        
        # 3. Extract the indices of the top K% worst scores
        # We use argsort for performance on numpy arrays
        top_k_indices = scores.argsort()[:k_threshold]
        
        # Save original dataframe indices as a set for set operations
        current_anomalies = set(df.iloc[top_k_indices].index)
        anomaly_sets[n] = current_anomalies
        
        # 4. Calculate Intersection over Union (IoU) with the previous step
        if n > components_range.start:
            prev_n = n - components_range.step
            prev_anomalies = anomaly_sets[prev_n]
            
            intersection = current_anomalies & prev_anomalies
            union = current_anomalies | prev_anomalies
            iou = len(intersection) / len(union) if union else 0.0
            
            stability_scores.append({"n_components": n, "IoU_vs_previous": iou})

    return pd.DataFrame(stability_scores)

# Example usage: check step-wise from 5 to 30 components
# stability_df = optimize_n_components_by_stability(df, components_range=range(5, 31, 5))
# print(stability_df)