import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import skfuzzy as fuzz
from datetime import datetime
from streamlit_option_menu import option_menu

# ==============================================================================
# 🌟 PERUBAHAN TAMPILAN WEB (UI/UX)
# ==============================================================================

# 1. Page Configuration: Menggunakan layout wide untuk tampilan lebih luas dan tema gelap (jika diaktifkan oleh user di setting Streamlit)
st.set_page_config(
    page_title='PERBANDINGAN METODE AHC DAN FCM UNTUK KLASTERISASI BATIK KABUPATEN BANGKALAN',
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Header
header_container = st.container()
with header_container:
    # Menggunakan kolom untuk memposisikan judul di tengah dengan layout wide
    col_logo, col_title, col_space = st.columns([1, 6, 1])
    with col_title:
        st.markdown(
            "<h1 style='text-align: center; color: #FF4B4B;'>PERBANDINGAN METODE AHC DAN FCM</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<h2 style='text-align: center; color: #F0F2F6; margin-top: -15px;'>UNTUK KLASTERISASI BATIK KABUPATEN BANGKALAN</h2>",
            unsafe_allow_html=True
        )
st.write("---") # PENGGANTI st.write("---") untuk pemisah

# 3. Menu Opsi: Mengubah warna menu
selected = option_menu(
    menu_title=None,
    options=["Deskripsi", "Preprocessing", "Pembobotan Entropy", "Clustering", "Implementasi"],
    icons=["house-door-fill", "calculator-fill", "clipboard-data-fill", "bar-chart-fill", "send-fill"],
    menu_icon=None,
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0E1117"},
        "icon": {"color": "orange", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#31333F"},
        "nav-link-selected": {"background-color": "#FF4B4B", "color": "white"}, # Warna merah Streamlit
    }
)

# -------------------------
# Helper: entropy weighting
# -------------------------
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

# ==============================================================================
# KONTEN PER HALAMAN
# ==============================================================================

# -------------------------
# Description
# -------------------------
if selected == "Deskripsi":
    st.markdown("## Deskripsi Dataset 📄")
    st.write("Dataset yang digunakan adalah data **UMKM Batik Kabupaten Bangkalan**. Informasi detail tentang dataset dapat dilihat pada tabel di bawah ini.")
    
    # Menggunakan container untuk dataset agar terlihat lebih rapi
    with st.container(border=True):
        try:
            df = pd.read_csv('DATA BATIK DINAS UMKM 1.csv')
            st.dataframe(df, use_container_width=True)
            st.markdown(f"**Total Data:** {len(df)} baris.")
        except FileNotFoundError:
            st.error("File 'DATA BATIK DINAS UMKM 1.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
            df = None

    if df is not None:
        st.write("---")
        st.markdown("#### Sumber dan Keterangan Data")
        st.markdown("👉 **Sumber Dataset:** Dinas Usaha UMKM Kabupaten Bangkalan")
        st.markdown("💡 **Deskripsi:** Dataset ini berisi informasi tentang UMKM batik, seperti lama usaha, aset, omzet, jumlah tenaga kerja, dan status surat izin.")

# -------------------------
# Preprocessing
# -------------------------
if selected == "Preprocessing":
    st.markdown("## Preprocessing Data 🛠️")
    try:
        df = pd.read_csv('DATA BATIK DINAS UMKM 1.csv')
    except FileNotFoundError:
        st.error("File 'DATA BATIK DINAS UMKM 1.csv' tidak ditemukan. Tidak dapat melanjutkan Preprocessing.")
        st.stop()

    # Dataset Asli
    st.subheader("1. Dataset Asli")
    st.dataframe(df, use_container_width=True)

    st.write("---")
    
    # Hapus kolom tidak relevan
    st.subheader("2. Menghapus Kolom Tidak Relevan")
    df_new = df.drop(['nama_usaha', 'alamat', 'tahun'], axis=1)
    st.info("Kolom yang dihapus: `nama_usaha`, `alamat`, `tahun` karena bersifat identitas/non-kriteria klasterisasi.")
    st.dataframe(df_new, use_container_width=True)

    st.write("---")

    # Label encoding surat Izin
    st.subheader("3. Label Encoding Kolom 'Surat Izin'")
    num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
    mapping_df = pd.DataFrame(list(num.items()), columns=["Status Izin", "Nilai Encoding"])
    st.dataframe(mapping_df, hide_index=True)
    
    df_new['surat Izin'] = df_new['surat Izin'].map(num)
    st.markdown("Dataset setelah **Label Encoding**:")
    st.dataframe(df_new, use_container_width=True)

    st.write("---")

    # Normalisasi
    st.subheader("4. Normalisasi Data (Min-Max Scaler)")
    st.markdown("Tujuan: Mengubah rentang nilai data menjadi **0–1** agar rentang nilai antar fitur tidak terlalu jauh dan tidak ada fitur yang mendominasi.")
    
    scaler_norm = MinMaxScaler()
    X_norm = scaler_norm.fit_transform(df_new)
    df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)

    df_compare = pd.concat(
        [df_new.reset_index(drop=True), df_scaled],
        axis=1,
        keys=["Data Asli", "Data Normalisasi"]
    )
    st.dataframe(df_compare, use_container_width=True)

# -------------------------
# Entropy Weighting
# -------------------------
if selected == "Pembobotan Entropy":
    st.markdown("## Pembobotan Entropy ⚖️")
    
    try:
        df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
    except FileNotFoundError:
        st.error("File 'DATA BATIK DINAS UMKM 1.csv' tidak ditemukan. Tidak dapat melanjutkan Pembobotan Entropy.")
        st.stop()
        
    df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)
    num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
    df_new["surat Izin"] = df_new["surat Izin"].map(num)
    scaler_norm = MinMaxScaler()
    X_norm = scaler_norm.fit_transform(df_new)
    df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)

    weights_entropy = entropy_weighting(df_scaled.values)
    df_entropy_weighted = df_scaled * weights_entropy

    # Bobot Entropy
    st.subheader("1. Bobot Entropy untuk Setiap Fitur")
    st.markdown("Bobot ini menunjukkan tingkat kepentingan (informasi) dari setiap fitur. Nilai bobot yang lebih tinggi menunjukkan fitur tersebut lebih informatif untuk klasterisasi.")
    weight_df = pd.DataFrame({"Fitur": df_new.columns, "Bobot": weights_entropy}).sort_values(by="Bobot", ascending=False).reset_index(drop=True)
    st.dataframe(weight_df, hide_index=True, use_container_width=True)

    st.write("---")

    # Visualisasi Bobot
    st.subheader("2. Visualisasi Bobot Fitur")
    # Menggunakan kolom untuk tata letak visual
    col_vis, col_desc = st.columns([2, 1])

    with col_vis:
        heatmap_data = pd.DataFrame([weight_df["Bobot"].values], columns=weight_df["Fitur"])
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(
            heatmap_data, annot=True, cmap="YlOrRd", cbar=False, fmt=".3f", ax=ax,
            linewidths=0.5, linecolor='black'
        )
        ax.set_title("Visualisasi Bobot Fitur (Entropy Weighting)", fontsize=14, weight="bold")
        ax.set_yticks([])
        st.pyplot(fig)
    
    with col_desc:
        st.markdown("**5 Fitur Terbaik (Highest Weight):**")
        selected_features = weight_df["Fitur"].iloc[:5].tolist()
        for i, feature in enumerate(selected_features):
            st.markdown(f"**{i+1}.** {feature}")
        st.caption("Fitur-fitur ini akan digunakan dalam skenario *'Dengan Seleksi Fitur'*.")

    st.write("---")
    
    # Data Hasil Pembobotan
    st.subheader("3. Data Hasil Pembobotan Entropy")
    st.markdown("Data yang sudah dinormalisasi dikalikan dengan bobot entropi masing-masing fitur.")
    st.dataframe(df_entropy_weighted, use_container_width=True)


# -------------------------
# Clustering (skenario uji)
# -------------------------
if selected == "Clustering":
    st.markdown("## Skenario Uji Klasterisasi 🔬")
    st.markdown("Menguji 4 skenario: AHC vs FCM, masing-masing dengan dan tanpa seleksi fitur (Entropy Weighting).")
    
    # Load & Preprocessing data
    try:
        df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
    except FileNotFoundError:
        st.error("File 'DATA BATIK DINAS UMKM 1.csv' tidak ditemukan. Tidak dapat melanjutkan Clustering.")
        st.stop()
        
    df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)
    num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
    df_new["surat Izin"] = df_new["surat Izin"].map(num)
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(df_new)
    df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)
    weights_entropy = entropy_weighting(df_scaled.values)
    df_entropy_weighted = df_scaled * weights_entropy
    feature_ranking = pd.DataFrame(
        {"Feature": df_new.columns, "Weight": weights_entropy}
    ).sort_values(by="Weight", ascending=False).reset_index(drop=True)
    selected_features = feature_ranking["Feature"].iloc[:5].tolist()
    X_sub = df_entropy_weighted[selected_features].values
    range_n_clusters = range(2, 11)

    # Inisialisasi list skor
    scores_agg_norm, scores_fcm_norm = [], []
    scores_agg_weighted, scores_fcm_weighted = [], []

    # Perhitungan AHC dan FCM (dibiarkan di luar expander agar state-nya tetap saat berpindah menu)
    # AHC - Tanpa Seleksi Fitur
    for k in range_n_clusters:
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        agg_labels = agg.fit_predict(X_norm)
        scores_agg_norm.append(silhouette_score(X_norm, agg_labels))
    best_agg_score_norm = max(scores_agg_norm)
    best_agg_k_norm = range_n_clusters[scores_agg_norm.index(best_agg_score_norm)]

    # FCM - Tanpa Seleksi Fitur
    data_norm_T = X_norm.T
    for k in range_n_clusters:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data_norm_T, c=k, m=2, error=0.005, maxiter=1000, init=None, seed=42)
        fcm_labels = np.argmax(u, axis=0)
        scores_fcm_norm.append(silhouette_score(X_norm, fcm_labels))
    best_fcm_score_norm = max(scores_fcm_norm)
    best_fcm_k_norm = range_n_clusters[scores_fcm_norm.index(best_fcm_score_norm)]
    
    # AHC - Dengan Seleksi Fitur
    for k in range_n_clusters:
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        agg_labels = agg.fit_predict(X_sub)
        scores_agg_weighted.append(silhouette_score(X_sub, agg_labels))
    best_agg_score_weighted = max(scores_agg_weighted)
    best_agg_k_weighted = range_n_clusters[scores_agg_weighted.index(best_agg_score_weighted)]

    # FCM - Dengan Seleksi Fitur
    data_weighted_T = X_sub.T
    for k in range_n_clusters:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data_weighted_T, c=k, m=2, error=0.005, maxiter=1000, init=None, seed=42)
        fcm_labels = np.argmax(u, axis=0)
        scores_fcm_weighted.append(silhouette_score(X_sub, fcm_labels))
    best_fcm_score_weighted = max(scores_fcm_weighted)
    best_fcm_k_weighted = range_n_clusters[scores_fcm_weighted.index(best_fcm_score_weighted)]


    # Tampilan Skenario Uji per bagian menggunakan st.expander
    
    # =====================================================
    # 1. AHC - Tanpa Seleksi Fitur
    # =====================================================
    with st.expander("▶️ AHC - Tanpa Seleksi Fitur", expanded=False):
        st.markdown("#### Agglomerative Hierarchical Clustering (AHC) dengan Semua Fitur")
        fig1, ax1 = plt.subplots()
        ax1.plot(range_n_clusters, scores_agg_norm, marker='o', color='skyblue')
        ax1.set_title("AHC - Tanpa Seleksi Fitur", fontsize=12)
        ax1.set_xlabel("Jumlah Cluster (k)")
        ax1.set_ylabel("Silhouette Score")
        ax1.axvline(best_agg_k_norm, color='r', linestyle='--', label=f'Best K={best_agg_k_norm}')
        ax1.legend()
        st.pyplot(fig1)
        
        st.markdown("**Hasil Silhouette Score per K:**")
        col_res, col_best = st.columns([1, 1])
        with col_res:
            res_df = pd.DataFrame({'K': range_n_clusters, 'Silhouette Score': [f"{s:.4f}" for s in scores_agg_norm]})
            st.dataframe(res_df, hide_index=True, use_container_width=True)
        with col_best:
            st.success(f"Best K = **{best_agg_k_norm}**, Silhouette = **{best_agg_score_norm:.4f}**")

    st.markdown("---")
    
    # =====================================================
    # 2. FCM - Tanpa Seleksi Fitur
    # =====================================================
    with st.expander("▶️ FCM - Tanpa Seleksi Fitur", expanded=False):
        st.markdown("#### Fuzzy C-Means (FCM) dengan Semua Fitur")
        fig2, ax2 = plt.subplots()
        ax2.plot(range_n_clusters, scores_fcm_norm, marker='o', color='salmon')
        ax2.set_title("FCM - Tanpa Seleksi Fitur", fontsize=12)
        ax2.set_xlabel("Jumlah Cluster (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.axvline(best_fcm_k_norm, color='r', linestyle='--', label=f'Best K={best_fcm_k_norm}')
        ax2.legend()
        st.pyplot(fig2)
        
        st.markdown("**Hasil Silhouette Score per K:**")
        col_res, col_best = st.columns([1, 1])
        with col_res:
            res_df = pd.DataFrame({'K': range_n_clusters, 'Silhouette Score': [f"{s:.4f}" for s in scores_fcm_norm]})
            st.dataframe(res_df, hide_index=True, use_container_width=True)
        with col_best:
            st.success(f"Best K = **{best_fcm_k_norm}**, Silhouette = **{best_fcm_score_norm:.4f}**")
            
    st.markdown("---")

    # =====================================================
    # 3. AHC - Dengan Seleksi Fitur
    # =====================================================
    with st.expander("▶️ AHC - Dengan Seleksi Fitur", expanded=False):
        st.markdown(f"#### Agglomerative Hierarchical Clustering (AHC) dengan **5 Fitur Terbaik**")
        st.caption(f"Fitur terpilih: {', '.join(selected_features)}")
        fig3, ax3 = plt.subplots()
        ax3.plot(range_n_clusters, scores_agg_weighted, marker='o', color='lightgreen')
        ax3.set_title("AHC - Dengan Seleksi Fitur", fontsize=12)
        ax3.set_xlabel("Jumlah Cluster (k)")
        ax3.set_ylabel("Silhouette Score")
        ax3.axvline(best_agg_k_weighted, color='r', linestyle='--', label=f'Best K={best_agg_k_weighted}')
        ax3.legend()
        st.pyplot(fig3)
        
        st.markdown("**Hasil Silhouette Score per K:**")
        col_res, col_best = st.columns([1, 1])
        with col_res:
            res_df = pd.DataFrame({'K': range_n_clusters, 'Silhouette Score': [f"{s:.4f}" for s in scores_agg_weighted]})
            st.dataframe(res_df, hide_index=True, use_container_width=True)
        with col_best:
            st.success(f"Best K = **{best_agg_k_weighted}**, Silhouette = **{best_agg_score_weighted:.4f}**")
            
    st.markdown("---")

    # =====================================================
    # 4. FCM - Dengan Seleksi Fitur
    # =====================================================
    with st.expander("▶️ FCM - Dengan Seleksi Fitur", expanded=False):
        st.markdown(f"#### Fuzzy C-Means (FCM) dengan **5 Fitur Terbaik**")
        st.caption(f"Fitur terpilih: {', '.join(selected_features)}")
        fig4, ax4 = plt.subplots()
        ax4.plot(range_n_clusters, scores_fcm_weighted, marker='o', color='gold')
        ax4.set_title("FCM - Dengan Seleksi Fitur", fontsize=12)
        ax4.set_xlabel("Jumlah Cluster (k)")
        ax4.set_ylabel("Silhouette Score")
        ax4.axvline(best_fcm_k_weighted, color='r', linestyle='--', label=f'Best K={best_fcm_k_weighted}')
        ax4.legend()
        st.pyplot(fig4)
        
        st.markdown("**Hasil Silhouette Score per K:**")
        col_res, col_best = st.columns([1, 1])
        with col_res:
            res_df = pd.DataFrame({'K': range_n_clusters, 'Silhouette Score': [f"{s:.4f}" for s in scores_fcm_weighted]})
            st.dataframe(res_df, hide_index=True, use_container_width=True)
        with col_best:
            st.success(f"Best K = **{best_fcm_k_weighted}**, Silhouette = **{best_fcm_score_weighted:.4f}**")

    st.markdown("---")
    
    # =====================================================
    # Rekapitulasi Hasil
    # =====================================================
    st.subheader("📊 Rekapitulasi Hasil Akhir")
    df_results = pd.DataFrame({
        "Skenario": [
            "AHC Tanpa Seleksi Fitur",
            "FCM Tanpa Seleksi Fitur",
            "AHC Dengan Seleksi Fitur",
            "FCM Dengan Seleksi Fitur"
        ],
        "Best K": [
            best_agg_k_norm,
            best_fcm_k_norm,
            best_agg_k_weighted,
            best_fcm_k_weighted
        ],
        "Silhouette": [
            round(best_agg_score_norm, 4),
            round(best_fcm_score_norm, 4),
            round(best_agg_score_weighted, 4),
            round(best_fcm_score_weighted, 4)
        ]
    }).sort_values(by="Silhouette", ascending=False).reset_index(drop=True)
    
    # Highlight baris terbaik
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #38761D; color: white' if v else '' for v in is_max]

    st.dataframe(
        df_results.style.apply(highlight_max, subset=['Silhouette']), 
        hide_index=True, 
        use_container_width=True
    )
    
    # Visualisasi Hasil
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    methods = df_results["Skenario"]
    best_scores = df_results["Silhouette"]
    best_ks = df_results["Best K"]
    bars = ax5.bar(
        methods, best_scores,
        color=['#3399FF', '#FF6666', '#66CC66', '#FFCC66'],
        width=0.6,
        edgecolor='black'
    )
    for bar, score, k in zip(bars, best_scores, best_ks):
        ax5.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{score:.4f}\n(k={k})",
            ha='center', va='bottom', fontsize=10, weight='bold'
        )
    ax5.set_ylabel("Silhouette Score")
    ax5.set_title("Perbandingan Silhouette Score Terbaik (AHC vs FCM)", fontsize=14, weight='bold')
    plt.ylim(0, max(best_scores) * 1.1)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig5)

    # Simpan Hasil Terbaik ke session_state
    best_row = df_results.iloc[0] # Baris pertama karena sudah disortir
    st.session_state["best_method"] = best_row["Skenario"]
    st.session_state["best_k"] = best_row["Best K"]
    st.session_state["best_score"] = best_row["Silhouette"]
    st.balloons()
    st.success(
        f"🏆 Metode terbaik adalah **{st.session_state['best_method']}** "
        f"dengan K={st.session_state['best_k']} dan Silhouette={st.session_state['best_score']:.4f}"
    )

    st.write("---")
    
    # =====================================================
    # Generate final labels & Download
    # =====================================================
    st.subheader("📥 Hasil Klasterisasi Final & Download")

    metode_terbaik = st.session_state["best_method"]
    k_terbaik = st.session_state["best_k"]
    
    # Re-running clustering for final labels & centroids/centers
    labels_final = None
    if "AHC" in metode_terbaik:
        model_final = AgglomerativeClustering(n_clusters=k_terbaik, linkage='ward')
        if "Dengan Seleksi Fitur" in metode_terbaik:
            labels_final = model_final.fit_predict(X_sub)
            centroids = np.array([X_sub[labels_final == i].mean(axis=0) for i in range(k_terbaik)])
        else:
            labels_final = model_final.fit_predict(X_norm)
            centroids = np.array([X_norm[labels_final == i].mean(axis=0) for i in range(k_terbaik)])
        st.session_state["ahc_centroids"] = centroids
    elif "FCM" in metode_terbaik:
        if "Dengan Seleksi Fitur" in metode_terbaik:
            data_final_T = X_sub.T
        else:
            data_final_T = X_norm.T
            
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_final_T, c=k_terbaik, m=2, error=0.005, maxiter=1000, init=None, seed=42
        )
        labels_final = np.argmax(u, axis=0)
        st.session_state["fcm_cntr"] = cntr

    labels_final = np.array(labels_final) + 1  # ubah jadi 1-based

    df_hasil = df.copy()
    df_hasil["Cluster"] = labels_final
    df_hasil = df_hasil.sort_values(by="Cluster").reset_index(drop=True)
    st.session_state["df_clustered"] = df_hasil

    st.markdown(f"**Preview 15 Data Hasil Klasterisasi (Metode Terbaik: {metode_terbaik}, K={k_terbaik}):**")
    st.dataframe(df_hasil.head(15), use_container_width=True)

    csv_hasil = df_hasil.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Hasil Klasterisasi (CSV)",
        data=csv_hasil,
        file_name="hasil_clustering_final.csv",
        mime="text/csv",
        type="primary"
    )
    
# -------------------------
# Implementation (prediksi)
# -------------------------
if selected == "Implementasi":
    st.markdown("## Implementasi Prediksi Cluster 🚀")

    if "best_method" not in st.session_state:
        st.warning("⚠️ Silakan jalankan dulu menu **Clustering** untuk menentukan metode terbaik dan menyimpan model/centroid.")
    else:
        best_method = st.session_state["best_method"]
        best_k = st.session_state["best_k"]
        st.info(f"📌 Prediksi akan menggunakan metode terbaik: **{best_method}** (K={best_k}).")

        # Form input
        with st.form("form_batik", border=True):
            st.markdown("#### Masukkan Data UMKM Baru:")
            
            # Kolom untuk Nama, Alamat, Tahun Berdiri
            col_nama, col_alamat = st.columns(2)
            with col_nama:
                nama_usaha = st.text_input("Nama Usaha", help="Contoh: Batik Indah Madura")
            with col_alamat:
                tahun_berdiri = st.number_input("Tahun Berdiri", min_value=1900, max_value=datetime.now().year, step=1, value=datetime.now().year-5, help="Tahun UMKM mulai beroperasi.")
            
            alamat = st.text_area("Alamat", help="Alamat lengkap UMKM.")
            
            # Hitung lama usaha (otomatis)
            lama_usaha = datetime.now().year - tahun_berdiri
            st.text_input("Lama Usaha (Tahun)", value=str(lama_usaha), disabled=True, help="Otomatis dihitung dari Tahun Berdiri.")
            
            st.markdown("---")
            
            # Kolom untuk Data Numerik & Kategori
            col1, col2 = st.columns(2)
            with col1:
                kemitraan = st.number_input("Kemitraan", min_value=0, value=0, help="Jumlah mitra atau kerjasama yang dimiliki.")
                aset = st.number_input("Aset (Jutaan)", min_value=0, value=50, help="Nilai total aset UMKM dalam jutaan Rupiah.")
                omzet = st.number_input("Omzet (Ribuan) per bulan", min_value=0, value=10000, help="Rata-rata omzet per bulan dalam ribuan Rupiah.")
            
            with col2:
                jml_naker = st.number_input("Jumlah Tenaga Kerja", min_value=0, value=5, help="Total jumlah karyawan atau tenaga kerja.")
                izin = st.selectbox("Surat Izin", ("", "tidak memiliki", "proses pengurusan", "SIUP"), help="Status kepemilikan surat izin usaha.")
                st.write("") # Spacer

            submit_pred = st.form_submit_button("Prediksi Cluster", type="primary")

        if submit_pred:
            # 🔹 Validasi input (semua harus diisi)
            if (
                not nama_usaha or not alamat or izin == ""
                or tahun_berdiri == 0 or aset == 0 or omzet == 0 or jml_naker == 0
            ):
                st.error("⚠️ Semua kolom input harus diisi dengan nilai yang valid sebelum melakukan prediksi!")
            else:
                # Proses Prediksi
                try:
                    # mapping izin
                    izin_map = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}

                    # Buat input_data untuk klasterisasi
                    input_data = pd.DataFrame([{
                        "lama usaha": lama_usaha,
                        "kemitraan": kemitraan,
                        "aset (jutaan)": aset,
                        "omzet (ribuan) perbulan": omzet,
                        "jml_naker": jml_naker,
                        "surat Izin": izin_map[izin]
                    }])

                    # load dataset asli untuk normalisasi ulang (agar scaler ter-fit)
                    df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
                    df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)
                    num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
                    df_new["surat Izin"] = df_new["surat Izin"].map(num)

                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(df_new)
                    input_scaled = scaler.transform(input_data[df_new.columns])

                    cluster_label = None
                    # Prediksi Cluster
                    if "AHC" in best_method:
                        centroids = st.session_state.get("ahc_centroids")
                        if centroids is None:
                            st.error("Centroid AHC belum tersedia. Harap jalankan kembali menu Clustering.")
                            st.stop()
                        
                        # Jika menggunakan seleksi fitur, ambil hanya fitur terpilih
                        if "Dengan Seleksi Fitur" in best_method:
                            feature_ranking = pd.DataFrame({"Feature": df_new.columns, "Weight": entropy_weighting(scaler.fit_transform(df_new))}).sort_values(by="Weight", ascending=False).reset_index(drop=True)
                            selected_features = feature_ranking["Feature"].iloc[:5].tolist()
                            input_scaled_sub = scaler.transform(input_data[selected_features])
                            centroids_sub = st.session_state.get("ahc_centroids")
                            dists = np.linalg.norm(centroids_sub - input_scaled_sub, axis=1)
                        else:
                            dists = np.linalg.norm(centroids - input_scaled, axis=1)
                        cluster_label = int(np.argmin(dists)) + 1
                        
                    elif "FCM" in best_method:
                        cntr = st.session_state.get("fcm_cntr")
                        if cntr is None:
                            st.error("Center FCM belum tersedia. Harap jalankan kembali menu Clustering.")
                            st.stop()
                            
                        # Jika menggunakan seleksi fitur, ambil hanya fitur terpilih
                        if "Dengan Seleksi Fitur" in best_method:
                            feature_ranking = pd.DataFrame({"Feature": df_new.columns, "Weight": entropy_weighting(scaler.fit_transform(df_new))}).sort_values(by="Weight", ascending=False).reset_index(drop=True)
                            selected_features = feature_ranking["Feature"].iloc[:5].tolist()
                            input_scaled_sub = scaler.transform(input_data[selected_features])
                            u_pred = fuzz.cluster.cmeans_predict(input_scaled_sub.T, cntr, m=2, error=0.005, maxiter=1000)[0]
                        else:
                            u_pred = fuzz.cluster.cmeans_predict(input_scaled.T, cntr, m=2, error=0.005, maxiter=1000)[0]
                        cluster_label = int(np.argmax(u_pred, axis=0)[0]) + 1

                    # Tampilkan hasil
                    st.subheader("✅ Hasil Prediksi Cluster")
                    st.markdown(
                        f"UMKM **{nama_usaha}** diprediksi masuk ke **Cluster {cluster_label}** "
                        f"(Metode: {best_method}, K={best_k})"
                    )

                    # Simpan hasil ke session_state (optional)
                    if "df_clustered" in st.session_state:
                        df_clustered = st.session_state["df_clustered"].copy()

                        # tambahkan row prediksi
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
                        
                        st.write("---")
                        st.markdown("**Dataset + Data Prediksi Baru:**")
                        st.dataframe(df_clustered.tail(20), use_container_width=True) # Tampilkan 20 data terakhir
                        
                        # download button
                        csv_combined = df_clustered.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Hasil Prediksi Gabungan (CSV)", 
                            data=csv_combined, 
                            file_name="hasil_prediksi_gabungan.csv", 
                            mime="text/csv",
                            type="secondary"
                        )
                    else:
                        st.warning("⚠️ Data hasil clustering asli tidak ditemukan di session state. Download tidak tersedia.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")


# ==============================================================================
# FOOTER
# ==============================================================================
st.write("---")
col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    st.markdown("<p style='text-align: center; font-size: 10px;'>By Fahrurrohman Ibnu Irsad Argyanto</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 10px;'>© Copyright 2025.</p>", unsafe_allow_html=True)
