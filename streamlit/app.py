"""
Customer Segmentation Through Behavioral Clustering - Interactive Dashboard
Streamlit Web Application (Dark Mode Optimized)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode CSS Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.6rem;
    }
    .metric-card {
        background-color: #151b28;
        border: 1px solid #263043;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .cluster-card {
        background-color: #151b28;
        border: 1px solid #263043;
        border-left: 5px solid #38bdf8;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.25);
    }
    .cluster-card h4 {
        margin-top: 0;
        margin-bottom: 6px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    path = 'Mall_Customers.csv' if os.path.exists('Mall_Customers.csv') else 'dataset/Mall_Customers.csv'
    df = pd.read_csv(path)
    return df

df_raw = load_dataset()

# Sidebar Controls
st.sidebar.markdown("## ⚙️ Clustering Controls")

config_option = st.sidebar.radio(
    "1. Feature Configuration",
    ["Configuration A (Income & Spending)", "Configuration B (Age, Income & Spending)"],
    index=0
)

algo_option = st.sidebar.selectbox(
    "2. Clustering Algorithm",
    ["K-Means Clustering", "Agglomerative Hierarchical", "DBSCAN (Density-Based)"],
    index=0
)

if algo_option in ["K-Means Clustering", "Agglomerative Hierarchical"]:
    n_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=5, step=1)
else:
    eps = st.sidebar.slider("DBSCAN Epsilon (eps)", min_value=0.20, max_value=0.80, value=0.35, step=0.05)
    min_samples = st.sidebar.slider("DBSCAN Min Samples", min_value=3, max_value=10, value=5, step=1)

# Navigation Section via Query Param or Sidebar Radio
section_param = st.query_params.get("section", "all")
sections_list = [
    "📑 Full Comprehensive View (All Sections)",
    "🎯 Clustering & Visualizations",
    "📊 Validation & Performance Metrics",
    "👥 Cluster Profiles & Personas",
    "💡 Business Recommendations",
    "📁 Dataset Explorer & Export"
]

default_idx = 0
if section_param == "clustering":
    default_idx = 1
elif section_param == "metrics":
    default_idx = 2
elif section_param == "profiles":
    default_idx = 3
elif section_param == "recommendations":
    default_idx = 4
elif section_param == "explorer":
    default_idx = 5
elif section_param == "all":
    default_idx = 0

view_mode = st.sidebar.radio(
    "3. Dashboard Section View",
    sections_list,
    index=default_idx
)

gender_filter = st.sidebar.multiselect("Filter by Gender (View Only)", ["Male", "Female"], default=["Male", "Female"])

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip:** K-Means with Configuration A at $k=5$ offers peak internal separation (Silhouette: 0.5547).")

# Header Section
st.markdown('<div class="main-title">🛍️ Customer Segmentation Through Behavioral Clustering</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unsupervised behavioral profiling and strategic customer segmentation for shopping mall marketing intelligence.</div>', unsafe_allow_html=True)

# Overview KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", len(df_raw))
with col2:
    st.metric("Avg Annual Income", f"${df_raw['Annual Income (k$)'].mean():.1f}k")
with col3:
    st.metric("Avg Spending Score", f"{df_raw['Spending Score (1-100)'].mean():.1f} / 100")
with col4:
    st.metric("Gender Ratio (F / M)", f"{(df_raw['Gender']=='Female').sum()} / {(df_raw['Gender']=='Male').sum()}")

# Machine Learning Pipeline
if "Configuration A" in config_option:
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
else:
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_raw[features])

if algo_option == "K-Means Clustering":
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=25)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
elif algo_option == "Agglomerative Hierarchical":
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
else:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    core_mask = labels != -1
    n_unique = len(set(labels)) - (1 if -1 in labels else 0)
    if n_unique > 1 and len(set(labels[core_mask])) > 1:
        sil = float(silhouette_score(X_scaled[core_mask], labels[core_mask]))
        db = float(davies_bouldin_score(X_scaled[core_mask], labels[core_mask]))
        ch = float(calinski_harabasz_score(X_scaled[core_mask], labels[core_mask]))
    else:
        sil, db, ch = 0.0, 0.0, 0.0

df_display = df_raw.copy()
df_display['Cluster'] = labels

# Dark Theme Palette
dark_palette = ['#38bdf8', '#4ade80', '#fb923c', '#f43f5e', '#a855f7', '#facc15', '#2dd4bf', '#ec4899', '#6366f1', '#e2e8f0']

show_all = "Full Comprehensive View" in view_mode

# ----------------- SECTION 1: Clustering & Visualizations -----------------
if show_all or "Clustering & Visualizations" in view_mode:
    st.markdown("---")
    st.markdown("### 🎯 1. Clustering Results & Geometric Space")
    st.caption(f"Algorithm: **{algo_option}** | Features: **{', '.join(features)}** | Cluster Count: **{len(np.unique(labels))}**")
    
    col_plot1, col_plot2 = st.columns([3, 2])
    
    with col_plot1:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#151b28')
        
        unique_labels = np.unique(labels)
        for i, lbl in enumerate(unique_labels):
            mask = df_display['Cluster'] == lbl
            c_name = f"Cluster {lbl}" if lbl != -1 else "Noise / Outlier"
            point_color = '#64748b' if lbl == -1 else dark_palette[i % len(dark_palette)]
            ax.scatter(
                df_display.loc[mask, 'Annual Income (k$)'],
                df_display.loc[mask, 'Spending Score (1-100)'],
                label=f"{c_name} (n={mask.sum()})",
                s=70, alpha=0.9, edgecolor='#0b0f19', linewidth=0.7,
                color=point_color
            )
            
        if algo_option == "K-Means Clustering" and "Configuration A" in config_option:
            centers_orig = scaler.inverse_transform(model.cluster_centers_)
            ax.scatter(centers_orig[:, 0], centers_orig[:, 1], s=250, color='#fde047', marker='*', edgecolor='black', zorder=10, label='Centroids')
            
        ax.set_title(f"Income vs. Spending Score ({algo_option})", fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
        ax.set_xlabel("Annual Income (k$)", fontsize=10, color='#cbd5e1')
        ax.set_ylabel("Spending Score (1-100)", fontsize=10, color='#cbd5e1')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, color='#263043', linestyle='--', alpha=0.6)
        
        leg = ax.legend(loc="upper right", frameon=True, fontsize=8.5, facecolor='#151b28', edgecolor='#334155')
        for text in leg.get_texts():
            text.set_color('#f1f5f9')
            
        st.pyplot(fig)
        plt.close()
        
    with col_plot2:
        st.markdown("#### Segment Size Distribution")
        counts = df_display['Cluster'].value_counts().sort_index()
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4.2), dpi=150)
        fig_bar.patch.set_facecolor('#0b0f19')
        ax_bar.set_facecolor('#151b28')
        
        bar_colors = ['#64748b' if c == -1 else dark_palette[i % len(dark_palette)] for i, c in enumerate(counts.index)]
        bars = ax_bar.bar([f"C{c}" if c != -1 else "Noise" for c in counts.index], counts.values, color=bar_colors, edgecolor='#0b0f19', width=0.6)
        ax_bar.set_ylabel("Customer Count", fontsize=10, color='#cbd5e1')
        ax_bar.set_title("Customer Count per Cluster", fontsize=11, fontweight='bold', color='#f8fafc', pad=10)
        ax_bar.tick_params(colors='#94a3b8')
        for spine in ax_bar.spines.values():
            spine.set_color('#334155')
        ax_bar.grid(True, axis='y', color='#263043', linestyle='--', alpha=0.6)
        
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h}\n({h/len(df_raw)*100:.1f}%)",
                        ha='center', va='bottom', fontweight='bold', fontsize=8.5, color='#f8fafc')
                        
        ax_bar.set_ylim(0, max(counts.values) * 1.25)
        st.pyplot(fig_bar)
        plt.close()

# ----------------- SECTION 2: Validation Metrics -----------------
if show_all or "Validation & Performance Metrics" in view_mode:
    st.markdown("---")
    st.markdown("### 📊 2. Internal Cluster Validation Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Silhouette Score", f"{sil:.4f}", help="Higher is better (range -1 to 1). Peak at k=5 Config A is 0.5547")
    with c2:
        st.metric("Davies-Bouldin Index", f"{db:.4f}", help="Lower is better. Optimal at k=5 Config A is 0.5722")
    with c3:
        st.metric("Calinski-Harabasz Index", f"{ch:.2f}", help="Higher is better. Measures between-cluster to within-cluster dispersion")

    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        st.markdown("""
        #### Configuration A (Annual Income & Spending Score)
        - **$k=5$ (Optimal Selected Model):**
          - **Silhouette Score:** `0.5547` (Global Peak across all experiments)
          - **Davies-Bouldin Index:** `0.5722` (Global Minimum / Best Separation)
          - **Calinski-Harabasz Index:** `248.65`
          - **Geometric Structure:** Distinct 4-quadrant + dense center
        """)
    with col_bench2:
        st.markdown("""
        #### Configuration B (Age, Annual Income & Spending Score)
        - **$k=6$ (Peak for 3D Feature Space):**
          - **Silhouette Score:** `0.4284` (Significantly lower cluster tightness)
          - **Davies-Bouldin Index:** `0.8254` (Higher cluster boundary overlap)
          - **Calinski-Harabasz Index:** `135.10`
          - **Geometric Structure:** Continuous spatial dispersion
        """)

# ----------------- SECTION 3: Cluster Profiles & Personas -----------------
if show_all or "Cluster Profiles & Personas" in view_mode:
    st.markdown("---")
    st.markdown("### 👥 3. Cluster Demographic & Behavioral Personas")
    
    profile_df = df_display.groupby('Cluster').agg({
        'CustomerID': 'count',
        'Age': ['mean', 'median', 'min', 'max'],
        'Annual Income (k$)': ['mean', 'median', 'min', 'max'],
        'Spending Score (1-100)': ['mean', 'median', 'min', 'max']
    }).round(1)
    st.dataframe(profile_df, use_container_width=True)

    personas = [
        {"title": "Cluster 0: Balanced Middle-Income Shoppers", "tag": "40.5% of Patrons", "tag_color": "#38bdf8", "age": "42.7 yrs", "inc": "$55.3k", "spd": "49.5 / 100", "desc": "The primary traffic driver and foundational backbone of mall visitors. Moderate income and predictable, disciplined consumption habits across mid-tier retail, grocery, and family dining."},
        {"title": "Cluster 1: Affluent VIP Spenders", "tag": "19.5% of Patrons", "tag_color": "#4ade80", "age": "32.7 yrs", "inc": "$86.5k", "spd": "82.1 / 100", "desc": "The highest-margin customer cohort. Young to mid-career affluent professionals with substantial disposable income and high willingness to invest in designer brands, luxury dining, and exclusive experiences."},
        {"title": "Cluster 2: Frugal High-Earners / Untapped Savers", "tag": "17.5% of Patrons", "tag_color": "#fb923c", "age": "41.1 yrs", "inc": "$88.2k", "spd": "17.1 / 100", "desc": "Substantial financial capacity but minimal mall engagement. Predominantly male (54.3%) and mature. Highly selective spenders visiting for utility rather than leisure. Immense untapped commercial potential."},
        {"title": "Cluster 3: Youthful Impulsive Spenders", "tag": "11.0% of Patrons", "tag_color": "#f43f5e", "age": "25.3 yrs", "inc": "$25.7k", "spd": "79.4 / 100", "desc": "Young demographic (students and early-career) with modest earnings but extraordinarily high spending propensity. Highly driven by trends, social media, fast-fashion pop-ups, lifestyle cafes, and entertainment."},
        {"title": "Cluster 4: Prudent Budget Shoppers", "tag": "11.5% of Patrons", "tag_color": "#a855f7", "age": "45.2 yrs", "inc": "$26.3k", "spd": "20.9 / 100", "desc": "Price-sensitive and economically cautious older patrons. Visit mall strictly for staple goods, discount clearances, and essential services with minimal discretionary splurge."}
    ]
    for p in personas:
        st.markdown(f"""
        <div class="cluster-card" style="border-left-color: {p['tag_color']};">
            <h4 style="color:#f8fafc;">{p['title']} <span style="background-color: {p['tag_color']}25; color: {p['tag_color']}; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; margin-left: 8px;">{p['tag']}</span></h4>
            <p style="margin:4px 0 8px 0; color:#94a3b8; font-size: 0.95rem;"><b>Avg Age:</b> {p['age']} &nbsp;|&nbsp; <b>Avg Income:</b> {p['inc']} &nbsp;|&nbsp; <b>Avg Spending Score:</b> {p['spd']}</p>
            <p style="margin:0; color:#cbd5e1; line-height: 1.5;">{p['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- SECTION 4: Business Recommendations -----------------
if show_all or "Business Recommendations" in view_mode:
    st.markdown("---")
    st.markdown("### 💡 4. Targeted Marketing & Customer Engagement Playbook")
    
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("""
        #### 🌟 Cluster 1: Affluent VIP Spenders
        - **Objective:** Maximize customer lifetime value, enhance brand advocacy, and insulate against competing retail destinations.
        - **Actionable Initiatives:**
          - Implement invitation-only VIP lounge access, valet parking, and private shopping concierges.
          - Partner with luxury apparel, fine jewelry, and upscale dining tenants for exclusive seasonal preview galas.
          - Deliver personalized 1-on-1 concierge communication rather than generic coupon blasts.
        
        #### 📈 Cluster 2: Frugal High-Earners
        - **Objective:** Convert latent purchasing power into active mall revenue.
        - **Actionable Initiatives:**
          - Conduct exit and intercept surveys to diagnose missing tenant categories.
          - Introduce high-utility categories: premium fitness clubs, specialty electronics, automotive care, and specialty groceries.
          - Curate frictionless convenience services (reserved parking, bespoke consultations).
        """)
    with rec_col2:
        st.markdown("""
        #### 🚀 Cluster 3: Youthful Impulsive Spenders
        - **Objective:** Capture trending social spend and build long-term generational affinity.
        - **Actionable Initiatives:**
          - Launch digital marketing activations via Instagram and TikTok, alongside interactive AR installations.
          - Introduce student discount evenings, trendy pop-up food markets, cinema passes, and lifestyle cafes.
          - Deploy mobile stamp cards, contactless payments, and social referral rewards.

        #### 🛡️ Cluster 0 & 4: Balanced & Budget Shoppers
        - **Objective:** Maintain steady footfall and maximize basket conversion during key seasonal sales.
        - **Actionable Initiatives:**
          - Organize family weekend bundle promotions, back-to-school fairs, and seasonal clearance events.
          - Provide everyday mall loyalty points redeemable for dining vouchers or free parking.
        """)

# ----------------- SECTION 5: Dataset Explorer & Export -----------------
if show_all or "Dataset Explorer & Export" in view_mode:
    st.markdown("---")
    st.markdown("### 📁 5. Customer Dataset Explorer & Export")
    filtered_df = df_display[df_display['Gender'].isin(gender_filter)]
    st.dataframe(filtered_df, use_container_width=True)
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Segmented Customer Data (CSV)",
        data=csv_data,
        file_name="segmented_mall_customers.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("<small style='color:#64748b;'>Customer Segmentation Through Behavioral Clustering | Built with Streamlit, Scikit-Learn & Pandas (Dark Mode)</small>", unsafe_allow_html=True)
