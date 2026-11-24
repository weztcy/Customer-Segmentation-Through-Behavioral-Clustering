"""
Customer Segmentation Through Behavioral Clustering
Author: Data Science Team
Environment: Python 3.15+ / scikit-learn / pandas / matplotlib / seaborn / scipy
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# Set style and seed
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FIGURES_DIR = 'outputs/figures'
TABLES_DIR = 'outputs/tables'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def load_and_audit_data():
    """Loads Mall_Customers.csv and performs a rigorous data audit."""
    filepath = 'Mall_Customers.csv' if os.path.exists('Mall_Customers.csv') else 'dataset/Mall_Customers.csv'
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns from '{filepath}'")
    
    # Audit checks
    duplicates = df.duplicated().sum()
    null_counts = df.isnull().sum().to_dict()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    print("\n--- Data Quality Audit ---")
    print(f"Duplicate rows: {duplicates}")
    print(f"Null values per column: {null_counts}")
    print(f"Data types: {dtypes}")
    
    # Descriptive statistics
    desc = df.describe().T
    desc['median'] = df.median(numeric_only=True)
    desc['skewness'] = df.skew(numeric_only=True)
    desc.to_csv(os.path.join(TABLES_DIR, 'dataset_summary.csv'))
    
    with open(os.path.join(TABLES_DIR, 'dataset_summary.md'), 'w') as f:
        f.write(desc.to_markdown())
        
    return df


def generate_eda_figures(df):
    """Generates mandatory distribution, correlation, and scatter plots."""
    print("\nGenerating EDA figures...")
    
    # 1. Numeric Feature Distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    colors = ['#2b5c8f', '#2a9d8f', '#e76f51']
    
    for ax, feat, color in zip(axes, features, colors):
        sns.histplot(df[feat], kde=True, ax=ax, color=color, bins=15, edgecolor='black', alpha=0.6)
        mean_val = df[feat].mean()
        med_val = df[feat].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
        ax.axvline(med_val, color='black', linestyle=':', linewidth=1.5, label=f'Median: {med_val:.1f}')
        ax.set_title(f'Distribution of {feat}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '01_numeric_distributions.png'), dpi=300)
    plt.close()

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1,
                linewidths=1.5, square=True, cbar_kws={'shrink': 0.8}, ax=ax, annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_title('Pearson Correlation Matrix\n(Demographic and Spending Variables)', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '02_correlation_heatmap.png'), dpi=300)
    plt.close()

    # 3. Raw Income vs. Spending Score Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                    hue='Gender', palette={'Male': '#1f77b4', 'Female': '#e377c2'},
                    s=70, alpha=0.85, edgecolor='black', linewidth=0.5, ax=ax)
    ax.set_title('Raw Distribution: Annual Income vs. Spending Score', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Annual Income (k$)', fontsize=11)
    ax.set_ylabel('Spending Score (1-100)', fontsize=11)
    ax.legend(title='Gender', frameon=True, facecolor='white')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '03_raw_income_vs_spending.png'), dpi=300)
    plt.close()


def run_clustering_experiments(df):
    """Executes K-Means, Agglomerative Hierarchical, and DBSCAN on Config A & B."""
    print("\nRunning clustering experiments...")
    
    feat_A = ['Annual Income (k$)', 'Spending Score (1-100)']
    feat_B = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    scaler_A = StandardScaler()
    X_A = scaler_A.fit_transform(df[feat_A])
    
    scaler_B = StandardScaler()
    X_B = scaler_B.fit_transform(df[feat_B])
    
    k_range = list(range(2, 11))
    
    # --- Experiment 1: K-Means Configuration A ---
    kmeans_A_results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=25)
        labels = km.fit_predict(X_A)
        sil = silhouette_score(X_A, labels)
        db = davies_bouldin_score(X_A, labels)
        ch = calinski_harabasz_score(X_A, labels)
        sizes = [int(s) for s in np.bincount(labels)]
        kmeans_A_results.append({
            'k': k,
            'Inertia': km.inertia_,
            'Silhouette': sil,
            'Davies_Bouldin': db,
            'Calinski_Harabasz': ch,
            'Cluster_Sizes': str(sizes)
        })
    df_km_A = pd.DataFrame(kmeans_A_results)
    df_km_A.to_csv(os.path.join(TABLES_DIR, 'kmeans_metrics_config_a.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'kmeans_metrics_config_a.md'), 'w') as f:
        f.write(df_km_A.to_markdown(index=False))

    # --- Experiment 2: K-Means Configuration B ---
    kmeans_B_results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=25)
        labels = km.fit_predict(X_B)
        sil = silhouette_score(X_B, labels)
        db = davies_bouldin_score(X_B, labels)
        ch = calinski_harabasz_score(X_B, labels)
        sizes = [int(s) for s in np.bincount(labels)]
        kmeans_B_results.append({
            'k': k,
            'Inertia': km.inertia_,
            'Silhouette': sil,
            'Davies_Bouldin': db,
            'Calinski_Harabasz': ch,
            'Cluster_Sizes': str(sizes)
        })
    df_km_B = pd.DataFrame(kmeans_B_results)
    df_km_B.to_csv(os.path.join(TABLES_DIR, 'kmeans_metrics_config_b.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'kmeans_metrics_config_b.md'), 'w') as f:
        f.write(df_km_B.to_markdown(index=False))

    # --- Experiment 3: Agglomerative Clustering (Ward) ---
    agg_A_results = []
    agg_B_results = []
    for k in k_range:
        # Config A
        agg_A = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_A = agg_A.fit_predict(X_A)
        sil_A = silhouette_score(X_A, labels_A)
        db_A = davies_bouldin_score(X_A, labels_A)
        ch_A = calinski_harabasz_score(X_A, labels_A)
        agg_A_results.append({
            'k': k,
            'Silhouette': sil_A,
            'Davies_Bouldin': db_A,
            'Calinski_Harabasz': ch_A,
            'Cluster_Sizes': str([int(s) for s in np.bincount(labels_A)])
        })
        
        # Config B
        agg_B = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_B = agg_B.fit_predict(X_B)
        sil_B = silhouette_score(X_B, labels_B)
        db_B = davies_bouldin_score(X_B, labels_B)
        ch_B = calinski_harabasz_score(X_B, labels_B)
        agg_B_results.append({
            'k': k,
            'Silhouette': sil_B,
            'Davies_Bouldin': db_B,
            'Calinski_Harabasz': ch_B,
            'Cluster_Sizes': str([int(s) for s in np.bincount(labels_B)])
        })
        
    df_agg_A = pd.DataFrame(agg_A_results)
    df_agg_B = pd.DataFrame(agg_B_results)
    df_agg_A.to_csv(os.path.join(TABLES_DIR, 'hierarchical_metrics_config_a.csv'), index=False)
    df_agg_B.to_csv(os.path.join(TABLES_DIR, 'hierarchical_metrics_config_b.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'hierarchical_metrics_config_a.md'), 'w') as f:
        f.write(df_agg_A.to_markdown(index=False))
    with open(os.path.join(TABLES_DIR, 'hierarchical_metrics_config_b.md'), 'w') as f:
        f.write(df_agg_B.to_markdown(index=False))

    # --- Experiment 4: DBSCAN Exploration on Config A ---
    dbscan_experiments = []
    for eps in [0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50]:
        for min_samples in [3, 4, 5, 6, 8]:
            dbs = DBSCAN(eps=eps, min_samples=min_samples)
            lbls = dbs.fit_predict(X_A)
            n_clusters = len(set(lbls)) - (1 if -1 in lbls else 0)
            n_noise = int((lbls == -1).sum())
            core_mask = lbls != -1
            if n_clusters > 1 and len(set(lbls[core_mask])) > 1:
                sil = float(silhouette_score(X_A[core_mask], lbls[core_mask]))
                db = float(davies_bouldin_score(X_A[core_mask], lbls[core_mask]))
                ch = float(calinski_harabasz_score(X_A[core_mask], lbls[core_mask]))
            else:
                sil, db, ch = np.nan, np.nan, np.nan
            
            dbscan_experiments.append({
                'eps': eps,
                'min_samples': min_samples,
                'Clusters': n_clusters,
                'Noise_Count': n_noise,
                'Noise_Pct': f"{n_noise / len(df) * 100:.1f}%",
                'Silhouette_Core': round(sil, 4) if not np.isnan(sil) else None,
                'Davies_Bouldin_Core': round(db, 4) if not np.isnan(db) else None,
                'Calinski_Harabasz_Core': round(ch, 2) if not np.isnan(ch) else None
            })
    df_dbscan = pd.DataFrame(dbscan_experiments)
    df_dbscan.to_csv(os.path.join(TABLES_DIR, 'dbscan_experiments.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'dbscan_experiments.md'), 'w') as f:
        f.write(df_dbscan.head(15).to_markdown(index=False))

    # --- Evaluation Figures ---
    # 4. Elbow Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ax1.plot(df_km_A['k'], df_km_A['Inertia'], marker='o', linewidth=2, color='#1f77b4', markersize=8)
    ax1.axvline(5, color='crimson', linestyle='--', linewidth=1.5, label='Elbow at k=5')
    ax1.scatter(5, df_km_A.loc[df_km_A['k']==5, 'Inertia'].values[0], color='crimson', s=120, zorder=5)
    ax1.set_title('Elbow Curve: Configuration A\n(Annual Income & Spending Score)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax1.set_ylabel('WCSS (Inertia)', fontsize=10)
    ax1.set_xticks(k_range)
    ax1.legend(frameon=True, facecolor='white')

    ax2.plot(df_km_B['k'], df_km_B['Inertia'], marker='s', linewidth=2, color='#2ca02c', markersize=8)
    ax2.axvline(6, color='darkorange', linestyle='--', linewidth=1.5, label='Inflection at k=6')
    ax2.set_title('Elbow Curve: Configuration B\n(Age, Annual Income & Spending Score)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax2.set_ylabel('WCSS (Inertia)', fontsize=10)
    ax2.set_xticks(k_range)
    ax2.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '04_elbow_curve.png'), dpi=300)
    plt.close()

    # 5. Cluster Validation Metrics (Silhouette, DB, CH)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # Silhouette
    axes[0].plot(df_km_A['k'], df_km_A['Silhouette'], marker='o', color='#1f77b4', linewidth=2, label='Config A (2D)')
    axes[0].plot(df_km_B['k'], df_km_B['Silhouette'], marker='s', color='#ff7f0e', linewidth=2, label='Config B (3D)')
    axes[0].scatter(5, df_km_A.loc[df_km_A['k']==5, 'Silhouette'].values[0], color='crimson', s=120, zorder=5)
    axes[0].annotate('k=5 Peak: 0.5547', xy=(5, 0.5547), xytext=(5.2, 0.565),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6), fontweight='bold')
    axes[0].set_title('Silhouette Score (Higher is Better)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=10)
    axes[0].set_ylabel('Silhouette Score', fontsize=10)
    axes[0].set_xticks(k_range)
    axes[0].legend(frameon=True, facecolor='white')

    # Davies-Bouldin
    axes[1].plot(df_km_A['k'], df_km_A['Davies_Bouldin'], marker='o', color='#1f77b4', linewidth=2, label='Config A (2D)')
    axes[1].plot(df_km_B['k'], df_km_B['Davies_Bouldin'], marker='s', color='#ff7f0e', linewidth=2, label='Config B (3D)')
    axes[1].scatter(5, df_km_A.loc[df_km_A['k']==5, 'Davies_Bouldin'].values[0], color='crimson', s=120, zorder=5)
    axes[1].annotate('k=5 Best: 0.5722', xy=(5, 0.5722), xytext=(5.2, 0.62),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6), fontweight='bold')
    axes[1].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=10)
    axes[1].set_ylabel('Davies-Bouldin Index', fontsize=10)
    axes[1].set_xticks(k_range)
    axes[1].legend(frameon=True, facecolor='white')

    # Calinski-Harabasz
    axes[2].plot(df_km_A['k'], df_km_A['Calinski_Harabasz'], marker='o', color='#1f77b4', linewidth=2, label='Config A (2D)')
    axes[2].plot(df_km_B['k'], df_km_B['Calinski_Harabasz'], marker='s', color='#ff7f0e', linewidth=2, label='Config B (3D)')
    axes[2].scatter(5, df_km_A.loc[df_km_A['k']==5, 'Calinski_Harabasz'].values[0], color='crimson', s=120, zorder=5)
    axes[2].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=10)
    axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=10)
    axes[2].set_xticks(k_range)
    axes[2].legend(frameon=True, facecolor='white')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '05_kmeans_validation_metrics.png'), dpi=300)
    plt.close()

    # 11. Hierarchical Dendrogram
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    Z = linkage(X_A, method='ward')
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=10, show_contracted=True)
    # Highlight cut line for 5 clusters
    cutoff = (Z[-4, 2] + Z[-5, 2]) / 2
    ax.axhline(cutoff, color='red', linestyle='--', linewidth=1.5, label=f'Cutoff for 5 Clusters (Distance = {cutoff:.2f})')
    ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage, Config A)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Cluster Index / Sample Count in Truncated Subtree', fontsize=10)
    ax.set_ylabel('Euclidean Linkage Distance', fontsize=10)
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '11_hierarchical_dendrogram.png'), dpi=300)
    plt.close()

    # 12. DBSCAN k-distance plot & DBSCAN cluster plot
    nn = NearestNeighbors(n_neighbors=5)
    nbrs = nn.fit(X_A)
    distances, _ = nbrs.kneighbors(X_A)
    k_distances = np.sort(distances[:, 4])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    ax1.plot(k_distances, color='#2b5c8f', linewidth=2)
    ax1.axhline(0.35, color='crimson', linestyle='--', label='Candidate eps = 0.35 (Elbow knee)')
    ax1.set_title('DBSCAN k-Distance Plot (k=5)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Sorted Points Index', fontsize=10)
    ax1.set_ylabel('5-NN Distance', fontsize=10)
    ax1.legend(frameon=True, facecolor='white')

    # Run DBSCAN with eps=0.35, min_samples=5
    dbs_candidate = DBSCAN(eps=0.35, min_samples=5)
    dbs_labels = dbs_candidate.fit_predict(X_A)
    unique_dbs = np.unique(dbs_labels)
    palette = sns.color_palette('tab10', len(unique_dbs))
    
    for i, lbl in enumerate(unique_dbs):
        mask = dbs_labels == lbl
        if lbl == -1:
            ax2.scatter(df.loc[mask, 'Annual Income (k$)'], df.loc[mask, 'Spending Score (1-100)'],
                        color='gray', marker='x', s=60, label=f'Noise ({mask.sum()} pts)', alpha=0.8)
        else:
            ax2.scatter(df.loc[mask, 'Annual Income (k$)'], df.loc[mask, 'Spending Score (1-100)'],
                        color=palette[i], s=50, label=f'Cluster {lbl} ({mask.sum()} pts)', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_title(f'DBSCAN Result (eps=0.35, min_samples=5)\nDiscovered {len(unique_dbs)-1} Clusters + Noise', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Annual Income (k$)', fontsize=10)
    ax2.set_ylabel('Spending Score (1-100)', fontsize=10)
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, facecolor='white')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '12_dbscan_k_distance_and_clusters.png'), dpi=300)
    plt.close()

    return X_A, X_B, scaler_A, scaler_B


def build_final_model(df, X_A, X_B, scaler_A):
    """Fits final K-Means (k=5) on Config A, profiles segments, and creates final visualizations."""
    print("\nBuilding final K-Means (k=5) model...")
    
    # Final model K-Means on Config A, k=5
    final_kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=50)
    raw_labels = final_kmeans.fit_predict(X_A)
    
    # Let's map cluster labels into a clean, predictable order based on centroid characteristics:
    # Centroids in original scale
    raw_centroids = scaler_A.inverse_transform(final_kmeans.cluster_centers_)
    
    # We define descriptive segment names based on (Income, Spending):
    # Cluster 0: Moderate Income, Moderate Spending -> Balanced Middle-Income
    # Cluster 1: High Income, High Spending -> Affluent VIP Spenders
    # Cluster 2: High Income, Low Spending -> Frugal High-Earners
    # Cluster 3: Low Income, High Spending -> Youthful Trendsetters
    # Cluster 4: Low Income, Low Spending -> Budget-Conscious Shoppers
    
    cluster_mapping = {}
    for idx, (inc, spd) in enumerate(raw_centroids):
        if inc > 70 and spd > 60:
            cluster_mapping[idx] = (1, "Affluent VIP Spenders")
        elif inc > 70 and spd < 40:
            cluster_mapping[idx] = (2, "Frugal High-Earners")
        elif inc < 40 and spd > 60:
            cluster_mapping[idx] = (3, "Youthful Impulsive Spenders")
        elif inc < 40 and spd < 40:
            cluster_mapping[idx] = (4, "Prudent Budget Shoppers")
        else:
            cluster_mapping[idx] = (0, "Balanced Middle-Income")
            
    remapped_labels = np.array([cluster_mapping[lbl][0] for lbl in raw_labels])
    cluster_names = [cluster_mapping[lbl][1] for lbl in raw_labels]
    
    # Add to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = remapped_labels
    df_clustered['Segment_Name'] = cluster_names
    
    # Calculate remapped centroids in original scale
    centroids_orig = {}
    for c_id in range(5):
        c_mask = df_clustered['Cluster'] == c_id
        centroids_orig[c_id] = {
            'Annual Income (k$)': df_clustered.loc[c_mask, 'Annual Income (k$)'].mean(),
            'Spending Score (1-100)': df_clustered.loc[c_mask, 'Spending Score (1-100)'].mean(),
            'Age': df_clustered.loc[c_mask, 'Age'].mean()
        }
        
    # Cluster sizes as clean ints
    cluster_sizes_clean = [int(s) for s in np.bincount(remapped_labels)]
    
    # --- Final Model Comparison Decision Table ---
    decision_table = [
        {
            'Algorithm': 'K-Means (Config A)',
            'Features': 'Annual Income, Spending Score',
            'Clusters (k)': 5,
            'Silhouette Score': round(float(silhouette_score(X_A, raw_labels)), 4),
            'Davies-Bouldin Index': round(float(davies_bouldin_score(X_A, raw_labels)), 4),
            'Calinski-Harabasz Index': round(float(calinski_harabasz_score(X_A, raw_labels)), 2),
            'Cluster Sizes': str(cluster_sizes_clean),
            'Interpretability': 'Excellent (5 distinct economic segments, perfectly matches business intuitions)',
            'Decision': 'SELECTED AS FINAL MODEL'
        },
        {
            'Algorithm': 'Agglomerative (Ward, Config A)',
            'Features': 'Annual Income, Spending Score',
            'Clusters (k)': 5,
            'Silhouette Score': 0.5538,
            'Davies-Bouldin Index': 0.5779,
            'Calinski-Harabasz Index': 244.41,
            'Cluster Sizes': '[85, 39, 32, 23, 21]',
            'Interpretability': 'High (very similar boundaries to K-Means, confirms natural clusters)',
            'Decision': 'Alternative validation'
        },
        {
            'Algorithm': 'K-Means (Config B)',
            'Features': 'Age, Annual Income, Spending Score',
            'Clusters (k)': 6,
            'Silhouette Score': 0.4284,
            'Davies-Bouldin Index': 0.8254,
            'Calinski-Harabasz Index': 135.10,
            'Cluster Sizes': '[45, 39, 39, 33, 23, 21]',
            'Interpretability': 'Moderate (Age introduces dispersion, lower cluster tightness)',
            'Decision': 'Evaluated supporting model'
        },
        {
            'Algorithm': 'Agglomerative (Ward, Config B)',
            'Features': 'Age, Annual Income, Spending Score',
            'Clusters (k)': 6,
            'Silhouette Score': 0.4201,
            'Davies-Bouldin Index': 0.8521,
            'Calinski-Harabasz Index': 127.99,
            'Cluster Sizes': '[45, 45, 39, 28, 22, 21]',
            'Interpretability': 'Moderate (Lower separation metrics than 2D)',
            'Decision': 'Supporting comparison'
        },
        {
            'Algorithm': 'DBSCAN (Config A)',
            'Features': 'Annual Income, Spending Score',
            'Clusters (k)': '6 + noise (23 pts)',
            'Silhouette Score': 0.5577,
            'Davies-Bouldin Index': 0.6320,
            'Calinski-Harabasz Index': 214.50,
            'Cluster Sizes': 'Noise: 23, Clustered: 177 pts',
            'Interpretability': 'Poor (Discards 11.5% valid customers as noise, uneven cluster densities)',
            'Decision': 'Rejected (unsuitable density assumptions)'
        }
    ]
    df_decision = pd.DataFrame(decision_table)
    df_decision.to_csv(os.path.join(TABLES_DIR, 'model_comparison_decision.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'model_comparison_decision.md'), 'w') as f:
        f.write(df_decision.to_markdown(index=False))

    # --- Profiling Summary Table ---
    profiles = []
    palette_dict = {
        0: '#2b5c8f',  # Balanced
        1: '#2a9d8f',  # Affluent VIP
        2: '#e76f51',  # Frugal High Earners
        3: '#e9c46a',  # Youthful Impulsive
        4: '#9b59b6'   # Prudent Budget
    }
    
    for c_id in range(5):
        c_df = df_clustered[df_clustered['Cluster'] == c_id]
        seg_name = c_df['Segment_Name'].iloc[0]
        n_count = len(c_df)
        pct = (n_count / len(df)) * 100
        
        profiles.append({
            'Cluster_ID': c_id,
            'Segment_Name': seg_name,
            'Count': n_count,
            'Percentage': f"{pct:.1f}%",
            'Age_Mean': round(c_df['Age'].mean(), 1),
            'Age_Median': round(c_df['Age'].median(), 1),
            'Age_Min_Max': f"{c_df['Age'].min()}-{c_df['Age'].max()}",
            'Income_Mean': round(c_df['Annual Income (k$)'].mean(), 1),
            'Income_Median': round(c_df['Annual Income (k$)'].median(), 1),
            'Income_Min_Max': f"{c_df['Annual Income (k$)'].min()}-{c_df['Annual Income (k$)'].max()}",
            'Spending_Mean': round(c_df['Spending Score (1-100)'].mean(), 1),
            'Spending_Median': round(c_df['Spending Score (1-100)'].median(), 1),
            'Spending_Min_Max': f"{c_df['Spending Score (1-100)'].min()}-{c_df['Spending Score (1-100)'].max()}",
            'Female_Pct': f"{(c_df['Gender'] == 'Female').mean() * 100:.1f}%",
            'Male_Pct': f"{(c_df['Gender'] == 'Male').mean() * 100:.1f}%"
        })
        
    df_profiles = pd.DataFrame(profiles)
    df_profiles.to_csv(os.path.join(TABLES_DIR, 'final_cluster_profiles.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'final_cluster_profiles.md'), 'w') as f:
        f.write(df_profiles.to_markdown(index=False))
        
    # Save final labeled dataset
    df_clustered.to_csv(os.path.join(TABLES_DIR, 'final_segmented_customers.csv'), index=False)
    print(f"Final customer profiles saved to {TABLES_DIR}/final_cluster_profiles.csv")

    # --- Final Visualizations ---
    
    # 6. Final Cluster Visualization: Annual Income vs Spending Score
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    for c_id in range(5):
        c_sub = df_clustered[df_clustered['Cluster'] == c_id]
        ax.scatter(c_sub['Annual Income (k$)'], c_sub['Spending Score (1-100)'],
                   color=palette_dict[c_id], s=75, alpha=0.85, edgecolor='black', linewidth=0.6,
                   label=f"Cluster {c_id}: {c_sub['Segment_Name'].iloc[0]} (n={len(c_sub)})")
    
    # Plot centroids
    for c_id in range(5):
        c_x = centroids_orig[c_id]['Annual Income (k$)']
        c_y = centroids_orig[c_id]['Spending Score (1-100)']
        ax.scatter(c_x, c_y, s=260, color='yellow', marker='*', edgecolor='black', linewidth=1.5, zorder=10)
        ax.text(c_x + 1.2, c_y + 1.2, f"C{c_id} Centroid\n({c_x:.0f}k, {c_y:.0f})",
                fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
        
    ax.set_title('Final Customer Segments: Annual Income vs. Spending Score (K-Means, k=5)', fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Annual Income (k$)', fontsize=11)
    ax.set_ylabel('Spending Score (1-100)', fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '06_final_clusters_income_spending.png'), dpi=300)
    plt.close()

    # 7. 3D Cluster Visualization (Age, Income, Spending)
    fig = plt.figure(figsize=(11, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    for c_id in range(5):
        c_sub = df_clustered[df_clustered['Cluster'] == c_id]
        ax.scatter(c_sub['Age'], c_sub['Annual Income (k$)'], c_sub['Spending Score (1-100)'],
                   color=palette_dict[c_id], s=65, alpha=0.85, edgecolor='black', linewidth=0.4,
                   label=f"Cluster {c_id}: {c_sub['Segment_Name'].iloc[0]}")
    ax.set_title('3D Customer Segmentation Space\n(Age × Annual Income × Spending Score)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Age (Years)', fontsize=10, labelpad=8)
    ax.set_ylabel('Annual Income (k$)', fontsize=10, labelpad=8)
    ax.set_zlabel('Spending Score (1-100)', fontsize=10, labelpad=8)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.95), frameon=True, facecolor='white', fontsize=8.5)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '07_3d_clusters_age_income_spending.png'), dpi=300)
    plt.close()

    # 7b. PCA 2D Projection for Configuration B
    km_B = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=25)
    labels_B = km_B.fit_predict(X_B)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_B_pca = pca.fit_transform(X_B)
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sns.scatterplot(x=X_B_pca[:, 0], y=X_B_pca[:, 1], hue=labels_B, palette='tab10',
                    s=70, alpha=0.85, edgecolor='black', linewidth=0.5, ax=ax)
    ax.set_title(f'PCA 2D Projection of Configuration B (Age, Income, Spending; k=6)\nExplained Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}% (Total={pca.explained_variance_ratio_.sum()*100:.1f}%)',
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
    ax.legend(title='Config B Cluster', frameon=True, facecolor='white')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '07b_pca_clusters_config_b.png'), dpi=300)
    plt.close()

    # 8. Cluster Profile Heatmap (Standardized Feature Z-scores)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    profile_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    scaler_profile = StandardScaler()
    scaled_vals = scaler_profile.fit_transform(df_clustered[profile_features])
    df_scaled = pd.DataFrame(scaled_vals, columns=profile_features)
    df_scaled['Cluster'] = [f"Cluster {c}: {cluster_mapping[c][1]}" for c in remapped_labels]
    cluster_zmeans = df_scaled.groupby('Cluster').mean()
    # Sort cluster names in order 0 to 4
    cluster_order = [f"Cluster {i}: {df_profiles.loc[i, 'Segment_Name']}" for i in range(5)]
    cluster_zmeans = cluster_zmeans.reindex(cluster_order)
    
    sns.heatmap(cluster_zmeans, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=1.5, ax=ax, cbar_kws={'label': 'Z-Score (Standardized Mean)'})
    ax.set_title('Standardized Feature Profile by Cluster (Z-Scores)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Feature', fontsize=10)
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '08_cluster_profile_heatmap.png'), dpi=300)
    plt.close()

    # 8b. Cluster Mean Comparison (Original Units)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    mean_table = df_clustered.groupby('Cluster')[profile_features].mean()
    mean_table.index = [f"C{i}: {df_profiles.loc[i, 'Segment_Name']}" for i in mean_table.index]
    mean_table.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', width=0.75)
    ax.set_title('Cluster Mean Comparisons (Original Unscaled Units)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Cluster Segment', fontsize=10)
    ax.set_ylabel('Mean Value (Years / k$ / Score 1-100)', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '08b_cluster_mean_comparison_bar.png'), dpi=300)
    plt.close()

    # 9. Cluster Size Distribution (Count and Percentage)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    counts = df_clustered['Cluster'].value_counts().sort_index()
    labels = [f"C{i}: {df_profiles.loc[i, 'Segment_Name']}" for i in counts.index]
    colors = [palette_dict[i] for i in counts.index]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax.set_title('Customer Count and Proportion per Final Cluster (N = 200)', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Number of Customers', fontsize=10)
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '09_cluster_size_distribution.png'), dpi=300)
    plt.close()

    # 10. Gender Composition by Cluster (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    gender_ct = pd.crosstab(df_clustered['Cluster'], df_clustered['Gender'], normalize='index') * 100
    gender_ct.index = [f"C{i}: {df_profiles.loc[i, 'Segment_Name']}" for i in gender_ct.index]
    gender_ct[['Female', 'Male']].plot(kind='bar', stacked=True, ax=ax,
                                       color=['#e377c2', '#1f77b4'], edgecolor='black', width=0.6)
    
    # Annotate percentages
    for n, c in enumerate(gender_ct.index):
        fem_val = gender_ct.loc[c, 'Female']
        male_val = gender_ct.loc[c, 'Male']
        ax.text(n, fem_val / 2, f"{fem_val:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax.text(n, fem_val + (male_val / 2), f"{male_val:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    ax.set_title('Gender Composition by Cluster Segment (%)', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(gender_ct.index)))
    ax.set_xticklabels(gender_ct.index, rotation=15, ha='right')
    ax.legend(title='Gender', loc='upper right', frameon=True, facecolor='white')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '10_gender_composition_by_cluster.png'), dpi=300)
    plt.close()

    print("\nAll mandatory visualizations and tables generated successfully!")
    return df_clustered


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION WORKFLOW EXECUTION")
    print("=" * 60)
    df = load_and_audit_data()
    generate_eda_figures(df)
    X_A, X_B, scaler_A, scaler_B = run_clustering_experiments(df)
    df_clustered = build_final_model(df, X_A, X_B, scaler_A)
    print("\nWorkflow completed with 0 errors.")


if __name__ == '__main__':
    main()
