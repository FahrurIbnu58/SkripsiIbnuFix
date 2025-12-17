import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import skfuzzy as fuzz
from datetime import datetime
from streamlit_option_menu import option_menu

# -------------------------------------
# 🌈 Styling Page
# -------------------------------------
st.set_page_config(
    page_title='PERBANDINGAN METODE AHC DAN FCM UNTUK KLASTERISASI BATIK KABUPATEN BANGKALAN',
    layout="wide",
    page_icon="🧵"
)

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(160deg, #f5f7fa 0%, #e4ebf5 100%);
        }

        h1 { color: #1f3c88 !important; font-weight: 700 !important; }
        h2, h3 { color: #1a237e !important; }

        hr {
            border: 1px solid #b0bec5 !important;
            margin-top: 10px;
            margin-bottom: 20px;
        }

        .stCard {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
            margin-bottom: 25px;
        }

        div.stButton > button:first-child {
            background-color: #1f77b4;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #105a8b;
            transform: translateY(-1px);
        }

        .stDownloadButton button {
            background: linear-gradient(90deg, #00b09b, #96c93d);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .stDownloadButton button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        .footer {
            text-align: center;
            font-size: 0.85rem;
            color: #607d8b;
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------
# Judul utama halaman
# -------------------------------------
st.markdown("""
<h1 style='text-align: center; font-size: 30px;'>
Perbandingan Metode Agglomerative Hierarchical Clustering (AHC) dan Fuzzy C-Means Clustering (FCM) Untuk Menentukan Klasterisasi UMKM Batik Di Kabupaten Bangkalan
</h1>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------------
# Menu Navigasi
# -------------------------------------
selected = option_menu(
    menu_title=None,
    options=["Description", "Preprocessing", "Entropy Weighting", "Clustering", "Implementation"],
    icons=["info-circle", "tools", "activity", "diagram-3", "cpu"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"background-color": "#ffffff", "padding": "0!important", "border-radius": "10px"},
        "icon": {"color": "#1f3c88", "font-size": "18px"},
        "nav-link": {
            "font-size": "14px",
            "font-weight": "600",
            "text-align": "center",
            "color": "#1f3c88",
            "padding": "10px 20px",
            "--hover-color": "#e8f0fe",
        },
        "nav-link-selected": {"background-color": "#1f3c88", "color": "white"},
    },
)

# ======================================================
# Helper function untuk Entropy Weighting
# ======================================================
def entropy_weighting(data):
    data = np.array(data, dtype=float)
    m, n = data.shape
    data = data - np.min(data, axis=0) + 1e-12
    col_sum = data.sum(axis=0) + 1e-12
    P = data / col_sum
    k = 1.0 / np.log(m)
    entropy = -k * (P * np.log(P + 1e-12)).sum(axis=0)
    weights = (1 - entropy) / (n - entropy.sum())
    return weights

# ======================================================
# Helper plot (rapi, tidak nabrak)
# ======================================================
def plot_metric_curve(range_n_clusters, values, best_k, title, ylabel, color="blue"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(list(range_n_clusters), values, marker='o', color=color)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel(ylabel)
    ax.axvline(best_k, color='red', linestyle='--', linewidth=1.5)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig

# ======================================================
# Description Section
# ======================================================
if selected == "Description":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>📘 DESKRIPSI DATASET</h2>", unsafe_allow_html=True)

        df = pd.read_csv('DATA BATIK DINAS UMKM 1.csv')
        st.dataframe(df, use_container_width=True)
        st.caption(f"📊 Dataset berisi {df.shape[0]} data UMKM Batik Kabupaten Bangkalan.")

        st.write("""
        Dataset yang digunakan merupakan data **UMKM Batik Kabupaten Bangkalan tahun 2025**
        yang diperoleh dari **Dinas Usaha UMKM Kabupaten Bangkalan**.  
        Dataset ini memuat informasi mengenai profil usaha batik di wilayah Bangkalan.

        **Fitur-fitur dalam dataset antara lain:**
        - 🏭 **Nama Usaha**
        - 📍 **Alamat**
        - 📅 **Tahun Berdiri**
        - ⏳ **Lama Usaha**
        - 🤝 **Kemitraan**
        - 💰 **Aset (jutaan)**
        - 📈 **Omzet (ribuan/bulan)**
        - 👥 **Jumlah Tenaga Kerja**
        - 📜 **Surat Izin**
        """)
        st.success("✅ Dataset berhasil dimuat dan siap digunakan.")
        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Preprocessing Section
# ======================================================
if selected == "Preprocessing":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>⚙️ PREPROCESSING DATA</h2>", unsafe_allow_html=True)

        df = pd.read_csv('DATA BATIK DINAS UMKM 1.csv')
        st.write("### 🔍 Data Asli")
        st.dataframe(df.head(), use_container_width=True)

        # Hapus kolom tidak relevan
        df_new = df.drop(['nama_usaha', 'alamat', 'tahun'], axis=1)
        st.subheader("Menghapus Kolom yang Tidak Relevan")
        st.write("Kolom yang dihapus: `nama_usaha`, `alamat`, `tahun`")
        st.dataframe(df_new)

        # Label encoding surat Izin
        st.subheader("Label Encoding Kolom 'surat Izin'")
        num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
        mapping_df = pd.DataFrame(list(num.items()), columns=["Value Asli", "Encoding"])
        st.dataframe(mapping_df)
        df_new['surat Izin'] = df_new['surat Izin'].map(num)
        st.write("Sesudah Encoding:")
        st.dataframe(df_new)

        # Normalisasi
        st.subheader("Normalisasi Data (0–1)")
        scaler_norm = MinMaxScaler()
        X_norm = scaler_norm.fit_transform(df_new)
        df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)

        df_compare = pd.concat(
            [df_new.reset_index(drop=True), df_scaled],
            axis=1,
            keys=["Data Asli", "Data Normalisasi"]
        )
        st.write("Perbandingan Data Asli vs Data Normalisasi (rentang 0–1)")
        st.dataframe(df_compare)

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Entropy Weighting Section
# ======================================================
if selected == "Entropy Weighting":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>📊 ENTROPY WEIGHTING</h2>", unsafe_allow_html=True)

        df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
        df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)

        num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
        df_new["surat Izin"] = df_new["surat Izin"].map(num)

        scaler_norm = MinMaxScaler()
        X_norm = scaler_norm.fit_transform(df_new)
        df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)

        weights_entropy = entropy_weighting(df_scaled.values)
        df_entropy_weighted = df_scaled * weights_entropy

        st.subheader("📌 Bobot Entropy untuk Setiap Fitur")
        weight_df = pd.DataFrame(
            {"Fitur": df_new.columns, "Bobot": weights_entropy}
        ).sort_values(by="Bobot", ascending=False).reset_index(drop=True)
        st.dataframe(weight_df)

        st.subheader("📊 Data Hasil Pembobotan Entropy")
        st.dataframe(df_entropy_weighted)

        st.subheader("🔥 Visualisasi Bobot Fitur (Entropy Weighting)")
        heatmap_data = pd.DataFrame([weight_df["Bobot"].values], columns=weight_df["Fitur"])
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", cbar=False, fmt=".3f", ax=ax)
        ax.set_title("Entropy Weighting - Feature Importance", fontsize=14, weight="bold")
        ax.set_yticks([])
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Clustering Section
# ======================================================
if selected == "Clustering":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>🔗 CLUSTERING: AHC vs FCM</h2>", unsafe_allow_html=True)

        st.write(""" Pada tahap ini Dilakukan Skenario Uji :
        1. AHC Tanpa Seleksi Fitur
        2. FCM Tanpa Seleksi Fitur
        3. AHC Dengan Seleksi Fitur
        4. FCM Dengan Seleksi Fitur
        """)

        # load & preprocessing
        df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
        df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)
        num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
        df_new["surat Izin"] = df_new["surat Izin"].map(num)

        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(df_new)
        df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)

        weights_entropy = entropy_weighting(df_scaled.values)
        df_entropy_weighted = df_scaled * weights_entropy

        # feature ranking
        feature_ranking = pd.DataFrame(
            {"Feature": df_new.columns, "Weight": weights_entropy}
        ).sort_values(by="Weight", ascending=False).reset_index(drop=True)
        selected_features = feature_ranking["Feature"].iloc[:5].tolist()

        range_n_clusters = range(2, 11)

        # =====================================================
        # 1. AHC - Tanpa Seleksi Fitur
        # =====================================================
        st.subheader("🔹 AHC - Tanpa Seleksi Fitur")
        scores_agg_norm, dbi_agg_norm = [], []

        for k in range_n_clusters:
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            agg_labels = agg.fit_predict(X_norm)

            sil = silhouette_score(X_norm, agg_labels)
            dbi = davies_bouldin_score(X_norm, agg_labels)

            scores_agg_norm.append(sil)
            dbi_agg_norm.append(dbi)

        best_agg_score_norm = max(scores_agg_norm)
        best_agg_k_norm = list(range_n_clusters)[scores_agg_norm.index(best_agg_score_norm)]

        best_agg_dbi_norm = min(dbi_agg_norm)
        best_agg_k_dbi_norm = list(range_n_clusters)[dbi_agg_norm.index(best_agg_dbi_norm)]

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_metric_curve(range_n_clusters, scores_agg_norm, best_agg_k_norm,
                                        "AHC (Tanpa Seleksi) - Silhouette", "Silhouette Score", color="blue"))
        with col2:
            st.pyplot(plot_metric_curve(range_n_clusters, dbi_agg_norm, best_agg_k_dbi_norm,
                                        "AHC (Tanpa Seleksi) - DBI", "Davies–Bouldin Index", color="black"))

        st.write("📌 Hasil per K:")
        for k, sil, dbi in zip(range_n_clusters, scores_agg_norm, dbi_agg_norm):
            st.text(f"k={k}, Silhouette={sil:.4f}, DBI={dbi:.4f}")

        st.info(f"✅ Best Silhouette: K={best_agg_k_norm}, Silhouette={best_agg_score_norm:.4f}")
        st.info(f"✅ Best DBI (Minimum): K={best_agg_k_dbi_norm}, DBI={best_agg_dbi_norm:.4f}")

        # =====================================================
        # 2. FCM - Tanpa Seleksi Fitur
        # =====================================================
        st.subheader("🔹 FCM - Tanpa Seleksi Fitur")
        scores_fcm_norm, dbi_fcm_norm = [], []

        data_norm_T = X_norm.T
        for k in range_n_clusters:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data_norm_T, c=k, m=2, error=0.005, maxiter=1000, init=None, seed=42
            )
            fcm_labels = np.argmax(u, axis=0)

            sil = silhouette_score(X_norm, fcm_labels)
            dbi = davies_bouldin_score(X_norm, fcm_labels)

            scores_fcm_norm.append(sil)
            dbi_fcm_norm.append(dbi)

        best_fcm_score_norm = max(scores_fcm_norm)
        best_fcm_k_norm = list(range_n_clusters)[scores_fcm_norm.index(best_fcm_score_norm)]

        best_fcm_dbi_norm = min(dbi_fcm_norm)
        best_fcm_k_dbi_norm = list(range_n_clusters)[dbi_fcm_norm.index(best_fcm_dbi_norm)]

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_metric_curve(range_n_clusters, scores_fcm_norm, best_fcm_k_norm,
                                        "FCM (Tanpa Seleksi) - Silhouette", "Silhouette Score", color="orange"))
        with col2:
            st.pyplot(plot_metric_curve(range_n_clusters, dbi_fcm_norm, best_fcm_k_dbi_norm,
                                        "FCM (Tanpa Seleksi) - DBI", "Davies–Bouldin Index", color="black"))

        st.write("📌 Hasil per K:")
        for k, sil, dbi in zip(range_n_clusters, scores_fcm_norm, dbi_fcm_norm):
            st.text(f"k={k}, Silhouette={sil:.4f}, DBI={dbi:.4f}")

        st.info(f"✅ Best Silhouette: K={best_fcm_k_norm}, Silhouette={best_fcm_score_norm:.4f}")
        st.info(f"✅ Best DBI (Minimum): K={best_fcm_k_dbi_norm}, DBI={best_fcm_dbi_norm:.4f}")

        # =====================================================
        # 3. AHC - Dengan Seleksi Fitur
        # =====================================================
        st.subheader("🔹 AHC - Dengan Seleksi Fitur (5 Fitur Teratas)")
        X_sub = df_entropy_weighted[selected_features].values

        scores_agg_weighted, dbi_agg_weighted = [], []
        for k in range_n_clusters:
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            agg_labels = agg.fit_predict(X_sub)

            sil = silhouette_score(X_sub, agg_labels)
            dbi = davies_bouldin_score(X_sub, agg_labels)

            scores_agg_weighted.append(sil)
            dbi_agg_weighted.append(dbi)

        best_agg_score_weighted = max(scores_agg_weighted)
        best_agg_k_weighted = list(range_n_clusters)[scores_agg_weighted.index(best_agg_score_weighted)]

        best_agg_dbi_weighted = min(dbi_agg_weighted)
        best_agg_k_dbi_weighted = list(range_n_clusters)[dbi_agg_weighted.index(best_agg_dbi_weighted)]

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_metric_curve(range_n_clusters, scores_agg_weighted, best_agg_k_weighted,
                                        "AHC (Seleksi) - Silhouette", "Silhouette Score", color="green"))
        with col2:
            st.pyplot(plot_metric_curve(range_n_clusters, dbi_agg_weighted, best_agg_k_dbi_weighted,
                                        "AHC (Seleksi) - DBI", "Davies–Bouldin Index", color="black"))

        st.write("📌 Hasil per K:")
        for k, sil, dbi in zip(range_n_clusters, scores_agg_weighted, dbi_agg_weighted):
            st.text(f"k={k}, Silhouette={sil:.4f}, DBI={dbi:.4f}")

        st.info(f"✅ Best Silhouette: K={best_agg_k_weighted}, Silhouette={best_agg_score_weighted:.4f}")
        st.info(f"✅ Best DBI (Minimum): K={best_agg_k_dbi_weighted}, DBI={best_agg_dbi_weighted:.4f}")

        # =====================================================
        # 4. FCM - Dengan Seleksi Fitur
        # =====================================================
        st.subheader("🔹 FCM - Dengan Seleksi Fitur (5 Fitur Teratas)")

        # Simpan df_entropy_weighted sekali agar stabil saat rerun
        if "df_entropy_weighted" not in st.session_state:
            st.session_state["df_entropy_weighted"] = df_entropy_weighted.copy()
        df_entropy_weighted = st.session_state["df_entropy_weighted"]

        k_feat = 5
        selected_features = feature_ranking["Feature"].iloc[:k_feat].tolist()
        removed_features = feature_ranking["Feature"].iloc[k_feat:].tolist()

        st.write(f"**Fitur digunakan:** {', '.join(selected_features)}")
        if removed_features:
            st.write(f"**Fitur tidak digunakan:** {', '.join(removed_features)}")

        X_sub = df_entropy_weighted[selected_features].values
        data_weighted_T = X_sub.T

        scores_fcm_weighted, dbi_fcm_weighted = [], []

        for k in range_n_clusters:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data_weighted_T, c=k, m=2, error=0.005, maxiter=1000, init=None, seed=42
            )
            fcm_labels = np.argmax(u, axis=0)

            sil = silhouette_score(X_sub, fcm_labels)
            dbi = davies_bouldin_score(X_sub, fcm_labels)

            scores_fcm_weighted.append(sil)
            dbi_fcm_weighted.append(dbi)

        best_fcm_score_weighted = max(scores_fcm_weighted)
        best_fcm_k_weighted = list(range_n_clusters)[scores_fcm_weighted.index(best_fcm_score_weighted)]

        best_fcm_dbi_weighted = min(dbi_fcm_weighted)
        best_fcm_k_dbi_weighted = list(range_n_clusters)[dbi_fcm_weighted.index(best_fcm_dbi_weighted)]

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_metric_curve(range_n_clusters, scores_fcm_weighted, best_fcm_k_weighted,
                                        "FCM (Seleksi) - Silhouette", "Silhouette Score", color="purple"))
        with col2:
            st.pyplot(plot_metric_curve(range_n_clusters, dbi_fcm_weighted, best_fcm_k_dbi_weighted,
                                        "FCM (Seleksi) - DBI", "Davies–Bouldin Index", color="black"))

        st.write("📊 **Hasil per K:**")
        for k, sil, dbi in zip(range_n_clusters, scores_fcm_weighted, dbi_fcm_weighted):
            st.text(f"k={k}, Silhouette={sil:.4f}, DBI={dbi:.4f}")

        st.info(f"✅ Best Silhouette: K={best_fcm_k_weighted}, Silhouette={best_fcm_score_weighted:.4f}")
        st.info(f"✅ Best DBI (Minimum): K={best_fcm_k_dbi_weighted}, DBI={best_fcm_dbi_weighted:.4f}")

        # Jalankan ulang FCM dengan K terbaik (berdasarkan Silhouette) untuk ringkasan cluster
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_weighted_T, c=best_fcm_k_weighted, m=2, error=0.005, maxiter=1000, init=None, seed=42
        )
        best_labels = np.argmax(u, axis=0) + 1

        df_with_cluster = df[selected_features].copy()
        df_with_cluster["Cluster"] = best_labels

        # Ringkasan cluster
        num_features = [col for col in selected_features if col != "surat Izin"]
        if num_features:
            cluster_min = df_with_cluster.groupby("Cluster")[num_features].min()
            cluster_max = df_with_cluster.groupby("Cluster")[num_features].max()
            cluster_ranges = cluster_min.astype(str) + " – " + cluster_max.astype(str)
            cluster_ranges = cluster_ranges.reset_index()
        else:
            cluster_ranges = pd.DataFrame({"Cluster": df_with_cluster["Cluster"].unique()})

        if "surat Izin" in selected_features:
            izin_dist = df_with_cluster.groupby(["Cluster", "surat Izin"]).size().unstack(fill_value=0).reset_index()
            cluster_summary = pd.merge(cluster_ranges, izin_dist, on="Cluster", how="left")
        else:
            cluster_summary = cluster_ranges

        st.write("📋 **Ringkasan Tiap Cluster dari FCM dengan seleksi fitur:**")
        st.dataframe(cluster_summary)

        # =====================================================
        # Rekapitulasi Hasil
        # =====================================================
        st.subheader("📊 Rekapitulasi Hasil 4 Skenario")

        df_results = pd.DataFrame({
            "Skenario": [
                "AHC Tanpa Seleksi Fitur",
                "FCM Tanpa Seleksi Fitur",
                "AHC Dengan Seleksi Fitur",
                "FCM Dengan Seleksi Fitur"
            ],
            "Best K (Silhouette)": [
                best_agg_k_norm,
                best_fcm_k_norm,
                best_agg_k_weighted,
                best_fcm_k_weighted
            ],
            "Silhouette (Max)": [
                round(best_agg_score_norm, 4),
                round(best_fcm_score_norm, 4),
                round(best_agg_score_weighted, 4),
                round(best_fcm_score_weighted, 4)
            ],
            "Best K (DBI)": [
                best_agg_k_dbi_norm,
                best_fcm_k_dbi_norm,
                best_agg_k_dbi_weighted,
                best_fcm_k_dbi_weighted
            ],
            "DBI (Min)": [
                round(best_agg_dbi_norm, 4),
                round(best_fcm_dbi_norm, 4),
                round(best_agg_dbi_weighted, 4),
                round(best_fcm_dbi_weighted, 4)
            ]
        })
        st.dataframe(df_results)

        # Grafik Silhouette (terpisah)
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        methods = df_results["Skenario"]
        best_scores = df_results["Silhouette (Max)"]
        best_ks = df_results["Best K (Silhouette)"]
        bars = ax5.bar(methods, best_scores, color=['skyblue', 'salmon', 'lightgreen', 'orange'], width=0.5)

        ymax = float(best_scores.max())
        ax5.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)

        for bar, score, k in zip(bars, best_scores, best_ks):
            ax5.text(bar.get_x() + bar.get_width()/2, score + (ymax * 0.03),
                     f"{score:.4f}\n(k={k})", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax5.set_ylabel("Silhouette Score (lebih besar lebih baik)")
        ax5.set_title("Perbandingan Silhouette Terbaik\nAHC vs FCM (Tanpa & Dengan Seleksi Fitur)", pad=15)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(fig5)

        # Grafik DBI (terpisah)
        fig_dbi, ax_dbi = plt.subplots(figsize=(10, 5))
        best_dbis = df_results["DBI (Min)"]
        best_ks_dbi = df_results["Best K (DBI)"]
        bars2 = ax_dbi.bar(methods, best_dbis, color=['skyblue', 'salmon', 'lightgreen', 'orange'], width=0.5)

        ymax2 = float(best_dbis.max())
        ax_dbi.set_ylim(0, ymax2 * 1.25 if ymax2 > 0 else 1)

        for bar, val, k in zip(bars2, best_dbis, best_ks_dbi):
            ax_dbi.text(bar.get_x() + bar.get_width()/2, val + (ymax2 * 0.03),
                        f"{val:.4f}\n(k={k})", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax_dbi.set_ylabel("DBI (lebih kecil lebih baik)")
        ax_dbi.set_title("Perbandingan DBI Terbaik (Minimum)\nAHC vs FCM (Tanpa & Dengan Seleksi Fitur)", pad=15)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(fig_dbi)

        # Pilih metode terbaik berdasarkan Silhouette (evaluasi utama)
        best_row = df_results.loc[df_results["Silhouette (Max)"].idxmax()]
        st.session_state["best_method"] = best_row["Skenario"]
        st.session_state["best_k"] = int(best_row["Best K (Silhouette)"])
        st.session_state["best_score"] = float(best_row["Silhouette (Max)"])

        st.success(
            f"✅ Metode terbaik adalah **{st.session_state['best_method']}** "
            f"dengan K={st.session_state['best_k']} dan Silhouette={st.session_state['best_score']:.4f}"
        )

        # =====================================================
        # Generate final labels & simpan hasil clustering ke session_state
        # =====================================================
        st.subheader("📥 Download Hasil Clustering")

        metode_terbaik = st.session_state["best_method"]
        k_terbaik = st.session_state["best_k"]

        labels_final = None

        if "AHC" in metode_terbaik:
            model_final = AgglomerativeClustering(n_clusters=k_terbaik, linkage='ward')
            labels_final = model_final.fit_predict(X_norm)
            centroids = np.array([X_norm[labels_final == i].mean(axis=0) for i in range(k_terbaik)])
            st.session_state["ahc_centroids"] = centroids
            st.session_state["ahc_model"] = model_final

        elif "FCM" in metode_terbaik:
            data_final_T = X_norm.T
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data_final_T, c=k_terbaik, m=2, error=0.005, maxiter=1000, init=None, seed=42
            )
            labels_final = np.argmax(u, axis=0)
            st.session_state["fcm_cntr"] = cntr

        labels_final = np.array(labels_final) + 1  # 1-based

        df_hasil = df.copy()
        df_hasil["Cluster"] = labels_final
        df_hasil = df_hasil.sort_values(by="Cluster").reset_index(drop=True)

        st.session_state["df_clustered"] = df_hasil

        st.dataframe(df_hasil)

        csv_hasil = df_hasil.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Hasil Clustering (CSV)",
            data=csv_hasil,
            file_name="hasil_clustering.csv",
            mime="text/csv"
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Implementation
# ======================================================
if selected == "Implementation":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'> IMPLEMENTATION DATA</h2>", unsafe_allow_html=True)

        if "best_method" not in st.session_state:
            st.warning("⚠️ Silakan jalankan dulu menu *Clustering* untuk menentukan metode terbaik.")
        else:
            best_method = st.session_state["best_method"]
            best_k = st.session_state["best_k"]
            st.info(f"📌 Prediksi menggunakan metode terbaik hasil clustering sebelumnya: **{best_method}** (K={best_k})")

            # Form input
            with st.form("form_batik"):
                nama_usaha = st.text_input("Nama Usaha")
                alamat = st.text_area("Alamat")
                tahun_berdiri = st.number_input("Tahun Berdiri", min_value=1900, max_value=datetime.now().year, step=1)

                lama_usaha = datetime.now().year - tahun_berdiri
                st.text_input("Lama Usaha (otomatis)", value=str(lama_usaha), disabled=True)

                kemitraan = st.number_input("Kemitraan", min_value=0)
                aset = st.number_input("Aset (Rp.)", min_value=0)
                omzet = st.number_input("Omzet per bulan (Rp.)", min_value=0)
                jml_naker = st.number_input("Jumlah Tenaga Kerja", min_value=0)
                izin = st.selectbox("Surat Izin", ("", "tidak memiliki", "proses pengurusan", "SIUP"))
                submit_pred = st.form_submit_button("Prediksi Cluster")

            if submit_pred:
                if (
                    not nama_usaha
                    or not alamat
                    or tahun_berdiri == 0
                    or kemitraan == 0
                    or aset == 0
                    or omzet == 0
                    or jml_naker == 0
                    or izin == ""
                ):
                    st.warning("⚠️ Semua input harus diisi sebelum melakukan prediksi!")
                else:
                    izin_map = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}

                    input_data = pd.DataFrame([{
                        "lama usaha": lama_usaha,
                        "kemitraan": kemitraan,
                        "aset (jutaan)": aset,
                        "omzet (ribuan) perbulan": omzet,
                        "jml_naker": jml_naker,
                        "surat Izin": izin_map[izin]
                    }])

                    df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
                    df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)
                    num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
                    df_new["surat Izin"] = df_new["surat Izin"].map(num)

                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(df_new)
                    input_data = input_data[df_new.columns]
                    input_scaled = scaler.transform(input_data)

                    if "ahc_centroids" not in st.session_state and "fcm_cntr" not in st.session_state:
                        if "AHC" in best_method:
                            model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
                            labels = model.fit_predict(X_scaled)
                            centroids = np.array([X_scaled[labels == i].mean(axis=0) for i in range(best_k)])
                            st.session_state["ahc_centroids"] = centroids
                        elif "FCM" in best_method:
                            data_T = X_scaled.T
                            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                                data_T, c=best_k, m=2, error=0.005, maxiter=1000, init=None, seed=42
                            )
                            st.session_state["fcm_cntr"] = cntr

                    cluster_label = None
                    if "AHC" in best_method:
                        centroids = st.session_state["ahc_centroids"]
                        dists = np.linalg.norm(centroids - input_scaled, axis=1)
                        cluster_label = int(np.argmin(dists)) + 1
                    elif "FCM" in best_method:
                        cntr = st.session_state["fcm_cntr"]
                        u_pred = fuzz.cluster.cmeans_predict(input_scaled.T, cntr, m=2, error=0.005, maxiter=1000)[0]
                        cluster_label = int(np.argmax(u_pred, axis=0)[0]) + 1

                    st.subheader("📊 Hasil Prediksi Cluster")
                    st.success(f"UMKM **{nama_usaha}** masuk ke **Cluster {cluster_label}** (Metode: {best_method}, K={best_k})")

                    if "df_clustered" not in st.session_state:
                        if "AHC" in best_method and "ahc_centroids" in st.session_state:
                            centroids = st.session_state["ahc_centroids"]
                            labels_all = np.argmin(np.linalg.norm(centroids[:, None, :] - X_scaled[None, :, :], axis=2), axis=0) + 1
                        elif "fcm_cntr" in st.session_state:
                            cntr = st.session_state["fcm_cntr"]
                            u_all = fuzz.cluster.cmeans_predict(X_scaled.T, cntr, m=2, error=0.005, maxiter=1000)[0]
                            labels_all = np.argmax(u_all, axis=0) + 1
                        else:
                            labels_all = np.zeros(X_scaled.shape[0], dtype=int) + 1

                        df_hasil = df.copy()
                        df_hasil["Cluster"] = labels_all
                        st.session_state["df_clustered"] = df_hasil

                    df_clustered = st.session_state["df_clustered"].copy()

                    new_row = {
                        "nama_usaha": nama_usaha,
                        "alamat": alamat,
                        "tahun": tahun_berdiri,
                        "lama usaha": lama_usaha,
                        "kemitraan": kemitraan,
                        "aset (jutaan)": aset,
                        "omzet (ribuan) perbulan": omzet,
                        "jml_naker": jml_naker,
                        "surat Izin": izin,
                        "Cluster": cluster_label
                    }
                    df_clustered = pd.concat([df_clustered, pd.DataFrame([new_row])], ignore_index=True, sort=False)
                    st.session_state["df_clustered"] = df_clustered

                    csv_combined = df_clustered.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Hasil Prediksi (CSV)",
                        data=csv_combined,
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Footer
# ======================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    By <b>Fahrurrohman Ibnu Irsad Argyanto</b><br>
    © 2025 — Klasterisasi UMKM Batik Bangkalan
</div>
""", unsafe_allow_html=True)
