import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Movies Data.csv"

def run_clustering(df, k):
    """Run KMeans clustering on the dataset with k clusters and visualize."""
    features = ['budget', 'votes', 'gross']
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    
    print(f"\n=== Clustering results for k={k} ===")
    print("Average feature values per cluster:")
    print(df_clustered.groupby('Cluster')[features].mean().round(2))
    print("\nCluster sizes:")
    print(df_clustered['Cluster'].value_counts().sort_index())
    
    # Visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        df_clustered['budget'], 
        df_clustered['votes'], 
        df_clustered['gross'], 
        c=df_clustered['Cluster'], 
        cmap='viridis',
        alpha=0.6,
        s=15
    )
    
    ax.set_title(f"3D Scatter of Movies by Budget, Votes, and Gross (k={k})")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Votes")
    ax.set_zlabel("Gross")
    
    # Add legend
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [f"Cluster {i}" for i in range(k)], title="Clusters", loc='best')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    cols = ["name", "genre", "budget", "votes", "score", "gross"]
    work = df[cols].copy()
    
    for col in ["budget", "votes", "score", "gross"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        
    work = work.dropna(subset=["budget", "votes", "gross"])
    
    run_clustering(work, k=2)
    run_clustering(work, k=4)

if __name__ == "__main__":
    main()
