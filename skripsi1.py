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

st.set_page_config(page_title='PERBANDINGAN METODE AHC DAN FCM UNTUK KLASTERISASI BATIK KABUPATEN BANGKALAN')
st.write("---")
st.markdown("<h1 style='text-align: center;'>PERBANDINGAN METODE AHC DAN FCM UNTUK KLASTERISASI BATIK KABUPATEN BANGKALAN</h1>", unsafe_allow_html=True)
st.write("---")

selected = option_menu(
    menu_title=None,
    options=["Description", "Preprocessing", "Entropy Weighting", "Clustering", "Implementation"],
    icons=["house","calculator","clipboard", "table", "send"],
    menu_icon=None,
    default_index=0,
    orientation="horizontal",
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

# -------------------------
# Description
# -------------------------
if selected == "Description":
    st.markdown("<h1 style='text-align: center;'>DESKRIPSI DATASET</h1>", unsafe_allow_html=True)
    st.write("###### Dataset yang digunakan adalah dataset UMKM Batik Kabupaten Bangkaln, dapat dilihat pada tabel dibawah ini:")
    df = pd.read_csv('DATA BATIK DINAS UMKM 1.csv')
    st.dataframe(df)
    st.write("###### Sumber Dataset : Dinas Usaha UMKM Kabupaten Bangkalan")
    st.write(" Dataset ini berisi informasi tentang UMKM batik yang berada di Kabupaten Bangkalan")
    

# -------------------------
# Preprocessing
# -------------------------
if selected == "Preprocessing":
    st.markdown("<h1 style='text-align: center;'>PREPROCESSING DATA</h1>", unsafe_allow_html=True)

    df = pd.read_csv('DATA BATIK DINAS UMKM 1.csv')
    st.subheader("Dataset Asli")
    st.dataframe(df)

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
    st.write("Perbandingan Data Asli vs Data Normalisasi, Mengubah rentan data menjadi 0-1 (agar rentan nilai antar fitur tidak terlalu jauh)")
    st.dataframe(df_compare)

    
# -------------------------
# Entropy Weighting
# -------------------------
if selected == "Entropy Weighting":
    st.markdown("<h1 style='text-align: center;'>ENTROPY WEIGHTING</h1>", unsafe_allow_html=True)

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
    weight_df = pd.DataFrame({"Fitur": df_new.columns, "Bobot": weights_entropy}).sort_values(by="Bobot", ascending=False).reset_index(drop=True)
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

    st.write("---")
    st.write("By Fahrurrohman Ibnu Irsad Argyanto")
    st.write("© Copyright 2025.")

# -------------------------
# Clustering (skenario uji)
# -------------------------
if selected == "Clustering":
    st.markdown("<h1 style='text-align: center;'>SKENARIO UJI</h1>", unsafe_allow_html=True)

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

    # =====================================================
    # 1. AHC - Tanpa Seleksi Fitur
    # =====================================================
    st.subheader("🔹 AHC - Tanpa Seleksi Fitur")
    range_n_clusters = range(2, 11)
    scores_agg_norm = []
    for k in range_n_clusters:
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        agg_labels = agg.fit_predict(X_norm)
        scores_agg_norm.append(silhouette_score(X_norm, agg_labels))
    best_agg_score_norm = max(scores_agg_norm)
    best_agg_k_norm = range_n_clusters[scores_agg_norm.index(best_agg_score_norm)]
    fig1, ax1 = plt.subplots()
    ax1.plot(range_n_clusters, scores_agg_norm, marker='o')
    ax1.set_title("AHC - Tanpa Seleksi Fitur")
    ax1.set_xlabel("Jumlah Cluster (k)")
    ax1.set_ylabel("Silhouette Score")
    ax1.axvline(best_agg_k_norm, color='r', linestyle='--')
    st.pyplot(fig1)
    st.write("📌 Hasil per K:")
    for k, score in zip(range_n_clusters, scores_agg_norm):
        st.text(f"k={k}, Silhouette={score:.4f}")
    st.info(f"Best K = {best_agg_k_norm}, Silhouette = {best_agg_score_norm:.4f}")

    # =====================================================
    # 2. FCM - Tanpa Seleksi Fitur
    # =====================================================
    st.subheader("🔹 FCM - Tanpa Seleksi Fitur")
    scores_fcm_norm = []
    data_norm_T = X_norm.T
    for k in range_n_clusters:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_norm_T, c=k, m=2, error=0.005, maxiter=1000, init=None, seed=42
        )
        fcm_labels = np.argmax(u, axis=0)
        scores_fcm_norm.append(silhouette_score(X_norm, fcm_labels))
    best_fcm_score_norm = max(scores_fcm_norm)
    best_fcm_k_norm = range_n_clusters[scores_fcm_norm.index(best_fcm_score_norm)]
    fig2, ax2 = plt.subplots()
    ax2.plot(range_n_clusters, scores_fcm_norm, marker='o', color='orange')
    ax2.set_title("FCM - Tanpa Seleksi Fitur")
    ax2.set_xlabel("Jumlah Cluster (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.axvline(best_fcm_k_norm, color='r', linestyle='--')
    st.pyplot(fig2)
    st.write("📌 Hasil per K:")
    for k, score in zip(range_n_clusters, scores_fcm_norm):
        st.text(f"k={k}, Silhouette={score:.4f}")
    st.info(f"Best K = {best_fcm_k_norm}, Silhouette = {best_fcm_score_norm:.4f}")

    # =====================================================
    # 3. AHC - Dengan Seleksi Fitur
    # =====================================================
    st.subheader("🔹 AHC - Dengan Seleksi Fitur")
    X_sub = df_entropy_weighted[selected_features].values
    scores_agg_weighted = []
    for k in range_n_clusters:
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        agg_labels = agg.fit_predict(X_sub)
        scores_agg_weighted.append(silhouette_score(X_sub, agg_labels))
    best_agg_score_weighted = max(scores_agg_weighted)
    best_agg_k_weighted = range_n_clusters[scores_agg_weighted.index(best_agg_score_weighted)]
    fig3, ax3 = plt.subplots()
    ax3.plot(range_n_clusters, scores_agg_weighted, marker='o', color='green')
    ax3.set_title("AHC - Dengan Seleksi Fitur")
    ax3.set_xlabel("Jumlah Cluster (k)")
    ax3.set_ylabel("Silhouette Score")
    ax3.axvline(best_agg_k_weighted, color='r', linestyle='--')
    st.pyplot(fig3)
    st.write("📌 Hasil per K:")
    for k, score in zip(range_n_clusters, scores_agg_weighted):
        st.text(f"k={k}, Silhouette={score:.4f}")
    st.info(f"Best K = {best_agg_k_weighted}, Silhouette = {best_agg_score_weighted:.4f}")

    # =====================================================
    # 4. FCM - Dengan Seleksi Fitur
    # =====================================================
    st.subheader("🔹 FCM - Dengan Seleksi Fitur")
    data_weighted_T = X_sub.T
    scores_fcm_weighted = []
    for k in range_n_clusters:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_weighted_T, c=k, m=2, error=0.005, maxiter=1000, init=None, seed=42
        )
        fcm_labels = np.argmax(u, axis=0)
        scores_fcm_weighted.append(silhouette_score(X_sub, fcm_labels))
    best_fcm_score_weighted = max(scores_fcm_weighted)
    best_fcm_k_weighted = range_n_clusters[scores_fcm_weighted.index(best_fcm_score_weighted)]
    fig4, ax4 = plt.subplots()
    ax4.plot(range_n_clusters, scores_fcm_weighted, marker='o', color='purple')
    ax4.set_title("FCM - Dengan Seleksi Fitur")
    ax4.set_xlabel("Jumlah Cluster (k)")
    ax4.set_ylabel("Silhouette Score")
    ax4.axvline(best_fcm_k_weighted, color='r', linestyle='--')
    st.pyplot(fig4)
    st.write("📌 Hasil per K:")
    for k, score in zip(range_n_clusters, scores_fcm_weighted):
        st.text(f"k={k}, Silhouette={score:.4f}")
    st.info(f"Best K = {best_fcm_k_weighted}, Silhouette = {best_fcm_score_weighted:.4f}")

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
    })
    st.dataframe(df_results)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    methods = df_results["Skenario"]
    best_scores = df_results["Silhouette"]
    best_ks = df_results["Best K"]
    bars = ax5.bar(
        methods, best_scores,
        color=['skyblue', 'salmon', 'lightgreen', 'orange'],
        width=0.5
    )
    for bar, score, k in zip(bars, best_scores, best_ks):
        ax5.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{score:.4f}\n(k={k})",
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    ax5.set_ylabel("Silhouette Score")
    ax5.set_title("Perbandingan Silhouette Score Tertinggi\nAHC vs FCM (Tanpa & Dengan Seleksi Fitur)")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha='right')
    st.pyplot(fig5)

    best_row = df_results.loc[df_results["Silhouette"].idxmax()]
    st.session_state["best_method"] = best_row["Skenario"]
    st.session_state["best_k"] = best_row["Best K"]
    st.session_state["best_score"] = best_row["Silhouette"]
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

    labels_final = np.array(labels_final) + 1  # ubah jadi 1-based

    # dataframe hasil clustering (tanpa Metode & K)
    df_hasil = df.copy()
    df_hasil["Cluster"] = labels_final
    df_hasil = df_hasil.sort_values(by="Cluster").reset_index(drop=True)

    # simpan ke session_state
    st.session_state["df_clustered"] = df_hasil

    # preview + download
    st.dataframe(df_hasil.head(15))

    csv_hasil = df_hasil.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Hasil Clustering (CSV)",
        data=csv_hasil,
        file_name="hasil_clustering.csv",
        mime="text/csv"
    )



# -------------------------
# Implementation (prediksi)
# -------------------------
if selected == "Implementation":
    st.markdown("<h1 style='text-align: center;'>IMPLEMENTASI</h1>", unsafe_allow_html=True)

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

            # Hitung lama usaha (otomatis)
            lama_usaha = datetime.now().year - tahun_berdiri
            st.text_input("Lama Usaha (otomatis)", value=str(lama_usaha), disabled=True)

            kemitraan = st.number_input("Kemitraan", min_value=0)
            aset = st.number_input("Aset", min_value=0)
            omzet = st.number_input("Omzet per bulan", min_value=0)
            jml_naker = st.number_input("Jumlah Tenaga Kerja", min_value=0)
            izin = st.selectbox("Surat Izin", ("", "tidak memiliki", "proses pengurusan", "SIUP"))  # kosong sebagai default
            submit_pred = st.form_submit_button("Prediksi Cluster")

        if submit_pred:
            # 🔹 Validasi input (semua harus diisi)
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
                # mapping izin
                izin_map = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}

                # buat input_data hanya kolom clustering
                input_data = pd.DataFrame([{
                    "lama usaha": lama_usaha,
                    "kemitraan": kemitraan,
                    "aset (jutaan)": aset,
                    "omzet (ribuan) perbulan": omzet,
                    "jml_naker": jml_naker,
                    "surat Izin": izin_map[izin]
                }])

                # load dataset asli
                df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
                df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)
                num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
                df_new["surat Izin"] = df_new["surat Izin"].map(num)

                # normalisasi
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(df_new)
                input_data = input_data[df_new.columns]
                input_scaled = scaler.transform(input_data)

                # model centroid
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

                # prediksi cluster
                cluster_label = None
                if "AHC" in best_method:
                    centroids = st.session_state["ahc_centroids"]
                    dists = np.linalg.norm(centroids - input_scaled, axis=1)
                    cluster_label = int(np.argmin(dists)) + 1
                elif "FCM" in best_method:
                    cntr = st.session_state["fcm_cntr"]
                    u_pred = fuzz.cluster.cmeans_predict(input_scaled.T, cntr, m=2, error=0.005, maxiter=1000)[0]
                    cluster_label = int(np.argmax(u_pred, axis=0)[0]) + 1

                # tampilkan hasil
                st.subheader("📊 Hasil Prediksi Cluster")
                st.success(f"UMKM **{nama_usaha}** masuk ke **Cluster {cluster_label}** (Metode: {best_method}, K={best_k})")

                # simpan hasil ke session_state
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

                # download button
                csv_combined = df_clustered.to_csv(index=False).encode("utf-8")
                st.download_button(label="📥 Download Hasil Prediksi (CSV)", data=csv_combined, file_name="hasil_prediksi.csv", mime="text/csv")





st.write("---")
st.write("By Fahrurrohman Ibnu Irsad Argyanto")
st.write("© Copyright 2025.")


