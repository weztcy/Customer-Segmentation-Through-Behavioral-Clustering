# Portfolio Project Data — Customer Segmentation Through Behavioral Clustering

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/customer_segmentation_through_behavioral/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image12.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "customer-segmentation-through-behavioral-clustering",
  "code": "PRO-26 // ML-11",
  "title": "Customer Segmentation Through Behavioral Clustering",
  "category": "machine-learning",
  "categoryLabel": "Unsupervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Customer-Segmentation-Through-Behavioral-Clustering",
  "license": "MIT License",
  "banner": "/images/projects/customer_segmentation_through_behavioral/banner.webp",
  "summary": "Unsupervised behavioral clustering pipeline segmenting 200 mall customers into five actionable personas with K-Means and Streamlit.",
  "projectDescription": "An unsupervised machine learning framework analyzing 200 mall customer profiles across K-Means, Agglomerative Hierarchical, and DBSCAN models. Evaluates feature configurations using internal validation indices and delivers five segments inside a dark-mode Streamlit dashboard.",
  "architectureSummary": "Mall_Customers.csv -> Data Quality Audit -> StandardScaler (Config A vs Config B) -> Multi-Algorithm Sweep (K-Means, Ward, DBSCAN) -> K-Means (k=5) Profiling & Streamlit Dashboard",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Streamlit",
    "Tabulate"
  ],
  "telemetryMatrix": {
    "projectType": "Unsupervised Clustering Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web Dashboard",
    "primaryDomain": "Retail Behavioral Analytics",
    "architecturePipeline": "StandardScaler -> K-Means (k=5) -> Profiles",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Unsupervised Clustering Pipeline",
    "learningParadigm": "Unsupervised Learning",
    "modelDomain": "Retail Customer Behavioral Analytics",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Behavioral Customer Segmentation",
      "Multi-Algorithm Cluster Validation",
      "Interactive Streamlit Dashboard"
    ]
  },
  "mlProblemContext": "Shopping malls face fierce e-commerce competition where untargeted mass promotions erode retail margins. Without predefined purchase labels in the 200-customer dataset, management requires an objective unsupervised clustering pipeline to uncover natural behavioral groups and guide targeted tenant marketing.",
  "mlDatasetOverview": {
    "profileBadge": "Tabular Retail Profiles",
    "source": "Mall Customers Dataset (dataset/original/Mall_Customers.csv)",
    "modality": "Tabular",
    "scale": "200 Customer Records (112 Female, 88 Male)",
    "inputShape": "200 x 2 (Config A) / 200 x 3 (Config B)",
    "dataFormat": "CSV",
    "description": "Contains CustomerID, Gender, Age (18-70), Annual Income ($15k-$137k), and Spending Score (1-99) with zero missing values and zero duplicates. Two $137k income entries flagged by 1.5x IQR were retained as valid affluent shoppers.",
    "provenanceNote": "MIT-licensed repository dataset audited programmatically in Python; evaluated on full standardized cohort without synthetic augmentation."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Behavioral & Demographic Space",
    "spaceNote": "CustomerID and Gender excluded from Euclidean distances; Gender and Age retained for post-hoc persona profiling.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Behavioral Cluster Assignment (C0-C4)",
      "type": "Unsupervised Partition (5 Segments)",
      "distribution": "C0: 40.5% (81), C1: 19.5% (39), C2: 17.5% (35), C4: 11.5% (23), C3: 11.0% (22)"
    },
    "inputSignals": [
      {
        "name": "Annual Income (k$)",
        "type": "Numerical",
        "importanceOrRole": "Primary Config A & B Axis (mean $60.56k)",
        "description": "Self-reported customer annual income ($15k-$137k), defining financial purchasing capacity across horizontal cluster quadrants."
      },
      {
        "name": "Spending Score (1-100)",
        "type": "Numerical",
        "importanceOrRole": "Primary Config A & B Axis (mean 50.20)",
        "description": "Mall-assigned spending propensity index (1-99) exhibiting trimodal density that separates active spenders from frugal visitors."
      },
      {
        "name": "Age (Years)",
        "type": "Numerical",
        "importanceOrRole": "Config B Axis & Post-Hoc Profiler (mean 38.85)",
        "description": "Customer age (18-70) showing moderate negative correlation with Spending Score (r = -0.327) and distinguishing youthful vs. prudent segments."
      },
      {
        "name": "Gender",
        "type": "Categorical",
        "importanceOrRole": "Post-Hoc Demographic Profiler (56% F / 44% M)",
        "description": "Binary demographic attribute excluded from Euclidean distance calculations to avoid distortion, used to profile segment gender skew."
      }
    ],
    "dataCharacteristicsSummary": "The 200-record dataset exhibits zero missing or duplicate entries across balanced Female (56.0%) and Male (44.0%) cohorts. Annual Income and Spending Score show near-zero linear correlation (r = 0.010) yet form five distinct geometric clusters, whereas Age correlates negatively with Spending Score (r = -0.327)."
  },
  "mlLearningApproach": {
    "paradigm": "Unsupervised Learning (Centroid, Hierarchical, and Density Clustering)",
    "methodologyDescription": "We benchmarked K-Means, Agglomerative Hierarchical (Ward linkage), and DBSCAN across k = 2 to 10 under two standardized feature sets: Configuration A (Income, Spending) and Configuration B (Age, Income, Spending). Internal metrics selected K-Means on Configuration A at k = 5 for peak cohesion and interpretability.",
    "justification": "Configuration A K-Means (k = 5) outperforms 3D Configuration B (Silhouette 0.5547 vs. 0.4284), aligns with Ward Hierarchical structure (Silhouette 0.5538), and preserves 100% customer coverage unlike DBSCAN which discards 11.5% of valid patrons as noise."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Data Quality & IQR Outlier Audit",
      "description": "Verified zero null values, zero duplicates, and monotonic CustomerID keys; audited Tukey 1.5x IQR bounds and retained two $137k affluent records.",
      "badge": "Quality Audit"
    },
    {
      "step": "02",
      "title": "Identifier & Categorical Exclusion",
      "description": "Removed CustomerID and binary Gender from Euclidean distance matrices to prevent synthetic scale distortion while retaining both for downstream profiling.",
      "badge": "Feature Isolation"
    },
    {
      "step": "03",
      "title": "Dual Feature Matrix Formulation",
      "description": "Constructed Configuration A (Income, Spending) and Configuration B (Age, Income, Spending) to empirically measure demographic impact on cluster cohesion.",
      "badge": "Config A vs B"
    },
    {
      "step": "04",
      "title": "Independent Z-Score Standardization",
      "description": "Fitted separate StandardScaler transformers (mean = 0, variance = 1) on X_A and X_B so Income, Spending Score, and Age contribute equally to Euclidean distance.",
      "badge": "StandardScaler"
    }
  ],
  "mlDataPreparationNote": "Scalers were fitted independently to Configuration A and Configuration B feature matrices prior to clustering, with random_state = 42 fixed globally for reproducibility.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "K-Means Clustering (Configuration A, k = 5)",
    "architectureSummary": "Lloyd-style centroid partitioning over standardized 2D Euclidean space (Annual Income, Spending Score) minimizing Within-Cluster Sum of Squares (WCSS), cross-validated against Ward Agglomerative Hierarchical clustering and DBSCAN.",
    "configurationBadge": "Deterministic Multi-Start",
    "hyperparameters": [
      {
        "name": "n_clusters",
        "value": "5",
        "description": "Selected optimal cluster count confirmed by WCSS elbow inflection and global peak Silhouette Score."
      },
      {
        "name": "n_init",
        "value": "50 (25 sweep)",
        "description": "Number of independent centroid initializations used in the final model fit to avoid suboptimal local minima."
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Fixed pseudo-random seed governing centroid initialization and PCA projections for full deterministic reproducibility."
      },
      {
        "name": "Feature Scaling",
        "value": "StandardScaler",
        "description": "Zero-mean unit-variance normalization applied to Annual Income and Spending Score before distance computation."
      }
    ],
    "objectiveCriterion": "Within-Cluster Sum of Squares (WCSS / Inertia)",
    "optimizationSolver": "Iterative Expectation-Maximization Centroid Updates (K-Means)",
    "trainingProcess": "Features were standardized to zero mean and unit variance using StandardScaler prior to clustering. The final K-Means model (k = 5, n_init = 50, random_state = 42) minimized within-cluster inertia to 65.57, converging reproducibly into five balanced partitions subsequently profiled in original unscaled units."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Multi-Metric Internal Cluster Validation & Cross-Algorithm Benchmark",
    "description": "Evaluated across k = 2 to 10 on both Configuration A and Configuration B using Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and Elbow Inertia, then cross-validated against Ward Hierarchical linkage and DBSCAN density sweeps.",
    "metrics": [
      {
        "label": "SILHOUETTE SCORE",
        "value": "0.5547",
        "benchmark": "Target > 0.50"
      },
      {
        "label": "DAVIES-BOULDIN INDEX",
        "value": "0.5722",
        "benchmark": "Target < 0.70"
      },
      {
        "label": "CALINSKI-HARABASZ",
        "value": "248.65",
        "benchmark": "Baseline > 200"
      },
      {
        "label": "WCSS INERTIA (K=5)",
        "value": "65.57",
        "benchmark": "Baseline < 108.92"
      },
      {
        "label": "WARD SILHOUETTE",
        "value": "0.5538",
        "benchmark": "Target > 0.50"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Dashboard: Clustering Results & Geometric Space",
      "badge": "Interactive Scatter & Bar",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image01.webp",
      "telemetryNote": "K-Means Config A (k=5) scatter plot with centroids and segment size bars (N=200)."
    },
    {
      "title": "Streamlit Dashboard: Internal Cluster Validation Metrics",
      "badge": "Validation Panel",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image02.webp",
      "telemetryNote": "Config A (k=5, Silhouette 0.5547) vs Config B (k=6, Silhouette 0.4284) benchmark."
    },
    {
      "title": "Streamlit Dashboard: Cluster Demographic & Behavioral Personas",
      "badge": "Persona Profiles",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image03.webp",
      "telemetryNote": "Unscaled summary statistics table and executive persona cards for discovered segments."
    },
    {
      "title": "Streamlit Dashboard: Targeted Marketing Engagement Playbook",
      "badge": "Strategy Playbook",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image04.webp",
      "telemetryNote": "Tailored retention, conversion, and loyalty campaigns mapped to Clusters 0 through 4."
    },
    {
      "title": "Streamlit Dashboard: Customer Dataset Explorer & CSV Export",
      "badge": "Data Explorer",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image05.webp",
      "telemetryNote": "Gender-filterable segmented customer records with direct CSV download support."
    },
    {
      "title": "DBSCAN k-Distance Knee Plot & Density Clustering Result",
      "badge": "Density Benchmark",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image06.webp",
      "telemetryNote": "5-NN knee at eps=0.35 yields 6 clusters plus 23 noise points (11.5% excluded)."
    },
    {
      "title": "Configuration B PCA 2D Projection & 3D Cluster Space",
      "badge": "PCA & 3D Space",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image07.webp",
      "telemetryNote": "PCA retains 77.6% variance (PC1 44.3%, PC2 33.3%) showing 3D boundary dispersion."
    },
    {
      "title": "Raw Gender Scatter vs. Final K-Means (k=5) Segments",
      "badge": "Cluster Partition",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image08.webp",
      "telemetryNote": "Demonstrates gender intermingling vs. five cleanly separated K-Means centroids."
    },
    {
      "title": "Numeric Feature Distributions & Elbow Curves (Config A vs B)",
      "badge": "EDA & Elbow",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image09.webp",
      "telemetryNote": "Trimodal Spending Score density and sharp WCSS elbow inflection at k=5 for Config A."
    },
    {
      "title": "Validation Curves, Correlation Matrix & Z-Score Heatmap",
      "badge": "Multi-Metric Suite",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image10.webp",
      "telemetryNote": "Global Silhouette peak (0.5547) and minimum Davies-Bouldin (0.5722) at k=5."
    },
    {
      "title": "Gender Composition & Hierarchical Ward Dendrogram",
      "badge": "Dendrogram & Mix",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image11.webp",
      "telemetryNote": "Ward linkage cutoff at distance 6.90 independently confirms five natural clusters."
    },
    {
      "title": "Original-Scale Cluster Mean Comparisons & Segment Counts",
      "badge": "Segment Profiling",
      "imageUrl": "/images/projects/customer_segmentation_through_behavioral/image12.webp",
      "telemetryNote": "Balanced segment distribution from 40.5% (n=81) down to 11.0% (n=22) across N=200."
    }
  ],
  "mlVisualizationTelemetryNote": "Ward Hierarchical linkage (Silhouette 0.5538) corroborates K-Means k=5 (Silhouette 0.5547), confirming that the five clusters reflect natural data geometry rather than centroid initialization artifacts.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Centroids partition patrons into four corner behavioral quadrants around a central middle-income core: Balanced Middle-Income (40.5%), Affluent VIP Spenders (19.5%), Frugal High-Earners (17.5%), Prudent Budget Shoppers (11.5%), and Youthful Impulsive Spenders (11.0%), enabling distinct tenant playbooks.",
    "keyDrivers": [
      {
        "factor": "Annual Income x Spending Score",
        "impact": "Defines four extreme quadrant personas and central core with 0.5547 Silhouette separation."
      },
      {
        "factor": "Cluster 1 (Affluent VIP Spenders)",
        "impact": "19.5% share (n=39) combining $86.5k mean income with 82.1 mean spending score at age 32.7."
      },
      {
        "factor": "Cluster 2 (Frugal High-Earners)",
        "impact": "17.5% share (n=35) holding highest mean income ($88.2k) yet lowest spend (17.1), 54.3% Male."
      },
      {
        "factor": "Age Post-Hoc Stratification",
        "impact": "Negative correlation with spend (r = -0.327) separates youthful spenders (25.3 yrs) from budget shoppers (45.2 yrs)."
      }
    ],
    "limitations": [
      {
        "title": "Small Cross-Sectional Sample (N = 200)",
        "description": "Dataset size suffices for macro behavioral segmentation but may miss rare micro-segments or seasonal shifts."
      },
      {
        "title": "Absence of Longitudinal RFM Logs",
        "description": "Lacks transaction timestamps, purchase recency, basket composition, and customer lifetime margin records."
      },
      {
        "title": "Opaque Spending Score Formula",
        "description": "Spending Score (1-100) is pre-aggregated by the mall without documented feature weights or dwell-time breakdown."
      }
    ],
    "futureImprovements": [
      {
        "title": "Integrate POS & Footfall RFM Data",
        "target": "Capture longitudinal Recency, Frequency, and Monetary dynamics"
      },
      {
        "title": "Track Segment Migration Over Time",
        "target": "Measure conversion of Cluster 2 savers into Cluster 1 VIP spenders"
      },
      {
        "title": "Layer Supervised Propensity Models",
        "target": "Predict campaign response and churn risk per discovered cluster"
      }
    ],
    "discoveryNote": "All clusters contain at least 22 patrons (11.0% minimum share), providing statistically viable cohorts for controlled A/B marketing pilots."
  },
  "mlModelArtifacts": [
    {
      "name": "tables/final_segmented_customers.csv",
      "format": "CSV"
    },
    {
      "name": "tables/final_cluster_profiles.csv",
      "format": "CSV"
    },
    {
      "name": "tables/model_comparison_decision.csv",
      "format": "CSV"
    },
    {
      "name": "tables/kmeans_metrics_config_a.csv",
      "format": "CSV"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Unsupervised Learning",
    "modelArchitecture": "K-Means Partitioning",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Internal Cluster Validation"
  }
}
```
