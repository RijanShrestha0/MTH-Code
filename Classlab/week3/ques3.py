import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Movies Data.csv"

def run_multiple_k3(df):
    """Run KMeans clustering multiple times with k=3 and different random initializations."""
    features = ['budget', 'votes', 'gross']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("K-Means (k=3) - 4 Different Random Initializations", fontsize=16)
    
    for i in range(4):
        kmeans = KMeans(n_clusters=3, init='random', n_init=1, random_state=i*42)
        labels = kmeans.fit_predict(X_scaled)
        
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.scatter(
            X['budget'], 
            X['votes'], 
            X['gross'], 
            c=labels, 
            cmap='viridis',
            alpha=0.6,
            s=15
        )
        
        ax.set_title(f"Run {i+1} (Random Seed: {i*42})")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Votes")
        ax.set_zlabel("Gross")
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Run {i+1} Cluster Sizes: {dict(zip(unique, counts))}")

    plt.tight_layout()
    plt.show()

def main():
    print("Loading data for Question 3...")
    df = pd.read_csv(DATA_FILE)
    
    cols = ["budget", "votes", "gross"]
    work = df[cols].copy()
    
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        
    work = work.dropna()
    
    run_multiple_k3(work)

if __name__ == "__main__":
    main()
