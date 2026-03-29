import pandas as pd
from skrub import GapEncoder, TableVectorizer
from sklearn.ensemble import IsolationForest
import numpy as np

def detect_with_gap_encoder(df: pd.DataFrame, n_topics: int = 10) -> pd.Series:
    """
    Converts categories into latent topics (continuous activations) and looks for anomalies.
    """
    # TableVectorizer applies GapEncoder to categorical columns
    # We set cardinality_threshold=1 to force GapEncoder on all categorical columns
    vectorizer = TableVectorizer(
        cardinality_threshold=1,
        high_cardinality=GapEncoder(n_components=n_topics, random_state=42)
    )
    
    # Transform categories into a dense matrix of continuous activations
    latent_topics = vectorizer.fit_transform(df)
    
    iso_forest = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    iso_forest.fit(latent_topics)
    
    # Extract anomaly scores (lower score = stronger anomaly)
    scores: np.ndarray = -iso_forest.decision_function(latent_topics)
    
    return pd.Series(scores, index=df.index, name="GapEncoder_Anomaly_Score")

# Example usage:
df_categorical = pd.DataFrame({
   'color': ['red', 'green', 'red', 'blue'],
   'size': ['Small', 'Medium', 'Large', 'Small'],
   'country': ['PL', 'DE', 'PL', 'US']
})

anomaly_scores = detect_with_gap_encoder(df_categorical, n_topics=2)

print(df_categorical)
print(anomaly_scores)