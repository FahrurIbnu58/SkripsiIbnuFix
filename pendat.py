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
from streamlit_option_menu import option_menu

st.set_page_config(page_title='PERBANDINGAN METODE AHC DAN FCM UNTUK KLASTERISASI BATIK KABUPATEN BANGKALAN')
st.write("---")
st.markdown("<h1 style='text-align: center;'>TUGAS AKHIR</h1>", unsafe_allow_html=True)
st.write("---")




selected = option_menu(
    menu_title=None,
    options=["Description", "Preprocessing", "Entropy Weighting", "Clustering", "Implementation"],
    icons=["house","calculator","clipboard", "table", "send"],
    menu_icon=None,
    default_index=0,
    orientation="horizontal",
)




if selected == "Description" :
    st.markdown("<h1 style='text-align: center;'>DESKRIPSI DATASET</h1>", unsafe_allow_html=True)
    st.write("###### Dataset yang digunakan adalah dataset UMKM Batik Kabupaten Bangkaln, dapat dilihat pada tabel dibawah ini:")
    df = pd.read_csv('DATA BATIK DINAS UMKM 1.csv')
    st.dataframe(df)
    st.write("###### Sumber Dataset : Dinas Usaha UMKM Kabupaten Bangkalan")
    st.write(" Dataset ini berisi informasi tentang UMKM batik yang berada di Kabupaten Bangkalan")
    st.write("---")
    st.write("By Fahrurrohman Ibnu Irsad Argyanto")
    st.write("© Copyright 2025.")

if selected == "Preprocessing":
    st.markdown("<h1 style='text-align: center;'>PREPROCESSING DATA</h1>", unsafe_allow_html=True)

    # Load dataset
    df = pd.read_csv('DATA BATIK DINAS UMKM 1.csv')
    st.subheader("Dataset Asli")
    st.dataframe(df)

    # === 1. Menghapus Variabel yang Tidak Relevan ===
    df_new = df.drop(['nama_usaha', 'alamat', 'tahun'], axis=1)
    st.subheader("Menghapus Kolom yang Tidak Relevan")
    st.write("Kolom yang dihapus: `nama_usaha`, `alamat`, `tahun`")
    st.dataframe(df_new)

    # === 2. Label Encoding Kolom 'surat Izin' ===
    st.subheader("Label Encoding Kolom 'surat Izin'")
    st.write("Sebelum Encoding:")

    num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}

    # Buat tabel mapping value -> angka encoding
    mapping_df = pd.DataFrame(list(num.items()), columns=["Value Asli", "Encoding"])
    st.dataframe(mapping_df)

    # Terapkan encoding ke kolom surat Izin
    df_new['surat Izin'] = df_new['surat Izin'].map(num)

    st.write("Sesudah Encoding:")
    st.dataframe(df_new)


    # === 3. Normalisasi Data (0–1) ===
    st.subheader("Normalisasi Data (0–1)")
    scaler_norm = MinMaxScaler()
    X_norm = scaler_norm.fit_transform(df_new)
    df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)

    # Gabungkan Data Asli dan Data Normalisasi dengan MultiIndex
    df_compare = pd.concat(
        [df_new.reset_index(drop=True), df_scaled],
        axis=1,
        keys=["Data Asli", "Data Normalisasi"]
    )

    st.write("Perbandingan Data Asli vs Data Normalisasi")
    st.dataframe(df_compare)

    st.write("---")
    st.write("By Fahrurrohman Ibnu Irsad Argyanto")
    st.write("© Copyright 2025.")





if selected == "Entropy Weighting":
    st.markdown("<h1 style='text-align: center;'>ENTROPY WEIGHTING</h1>", unsafe_allow_html=True)

    # === Fungsi Entropy Weighting ===
    def entropy_weighting(data):
        data = np.array(data, dtype=float)
        m, n = data.shape

        # Geser data ke positif (karena ada yang negatif setelah standardisasi)
        data = data - np.min(data, axis=0) + 1e-12

        # Normalisasi (proporsi tiap fitur 0-1)
        col_sum = data.sum(axis=0) + 1e-12
        P = data / col_sum  # bentuk probabilitas

        # Hitung entropi tiap fitur
        k = 1.0 / np.log(m)
        entropy = -k * (P * np.log(P + 1e-12)).sum(axis=0)

        # Hitung bobot (semakin kecil entropi → bobot makin besar)
        weights = (1 - entropy) / (n - entropy.sum())
        return weights

    # === Load dataset Batik ===
    df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")

    # Hapus kolom non-numerik
    df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)

    # Label Encoding kolom 'surat Izin'
    num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
    df_new["surat Izin"] = df_new["surat Izin"].map(num)

    # Normalisasi (0–1)
    scaler_norm = MinMaxScaler()
    X_norm = scaler_norm.fit_transform(df_new)
    df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)

    # === Hitung Bobot Entropy ===
    weights_entropy = entropy_weighting(df_scaled.values)

    # Terapkan bobot ke data normalisasi
    df_entropy_weighted = df_scaled * weights_entropy

    # === Tampilkan bobot entropy tiap fitur ===
    st.subheader("📌 Bobot Entropy untuk Setiap Fitur")
    weight_df = pd.DataFrame({
        "Fitur": df_new.columns,
        "Bobot": weights_entropy
    }).sort_values(by="Bobot", ascending=False).reset_index(drop=True)
    st.dataframe(weight_df)

    # === Tampilkan data hasil pembobotan entropy ===
    st.subheader("📊 Data Hasil Pembobotan Entropy")
    st.dataframe(df_entropy_weighted)

    # === Visualisasi Heatmap Bobot ===
    st.subheader("🔥 Visualisasi Bobot Fitur (Entropy Weighting)")
    heatmap_data = pd.DataFrame([weight_df["Bobot"].values],
                                columns=weight_df["Fitur"])

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", cbar=False, fmt=".3f", ax=ax)
    ax.set_title("Entropy Weighting - Feature Importance", fontsize=14, weight="bold")
    ax.set_yticks([])
    st.pyplot(fig)

    st.write("---")
    st.write("By Fahrurrohman Ibnu Irsad Argyanto")
    st.write("© Copyright 2025.")


if selected == "Clustering":
    st.markdown("<h1 style='text-align: center;'>SKENARIO UJI</h1>", unsafe_allow_html=True)

    # Load data
    df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
    df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)
    num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
    df_new["surat Izin"] = df_new["surat Izin"].map(num)

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(df_new)
    df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)

    # Fungsi entropy weighting
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

    weights_entropy = entropy_weighting(df_scaled.values)
    df_entropy_weighted = df_scaled * weights_entropy

    # Ranking fitur
    feature_ranking = pd.DataFrame({
        "Feature": df_new.columns,
        "Weight": weights_entropy
    }).sort_values(by="Weight", ascending=False).reset_index(drop=True)
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
        sil = silhouette_score(X_norm, agg_labels)
        scores_agg_norm.append(sil)

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
            data_norm_T, c=k, m=2, error=0.005, maxiter=1000, init=None
        )
        fcm_labels = np.argmax(u, axis=0)
        sil = silhouette_score(X_norm, fcm_labels)
        scores_fcm_norm.append(sil)

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
        sil = silhouette_score(X_sub, agg_labels)
        scores_agg_weighted.append(sil)

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
            data_weighted_T, c=k, m=2, error=0.005, maxiter=1000, init=None
        )
        fcm_labels = np.argmax(u, axis=0)
        sil = silhouette_score(X_sub, fcm_labels)
        scores_fcm_weighted.append(sil)

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
    # 5. Rekap & Perbandingan
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

    # Visualisasi perbandingan (bar chart)
    st.subheader("🔥 Visualisasi Perbandingan Silhouette Score Tertinggi")
    methods = df_results["Skenario"]
    best_scores = df_results["Silhouette"]
    best_ks = df_results["Best K"]

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    bars = ax5.bar(methods, best_scores,
                color=['skyblue', 'salmon', 'lightgreen', 'orange'], width=0.5)

    for bar, score, k in zip(bars, best_scores, best_ks):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{score:.4f}\n(k={k})", ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax5.set_ylabel("Silhouette Score")
    ax5.set_title("Perbandingan Silhouette Score Tertinggi\nAHC vs FCM (Tanpa & Dengan Seleksi Fitur)")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha='right')
    st.pyplot(fig5)

    # =====================================================
    # 6. Simpan Metode Terbaik ke session_state
    # =====================================================
    best_row = df_results.loc[df_results["Silhouette"].idxmax()]
    st.session_state["best_method"] = best_row["Skenario"]
    st.session_state["best_k"] = best_row["Best K"]
    st.session_state["best_score"] = best_row["Silhouette"]

    st.success(f"✅ Metode terbaik adalah **{st.session_state['best_method']}** "
            f"dengan K={st.session_state['best_k']} "
            f"dan Silhouette={st.session_state['best_score']:.4f}")

        # =====================================================
    # 7. Simpan hasil clustering ke CSV
    # =====================================================
    st.subheader("📥 Download Hasil Clustering")

    # Pilih metode terbaik
    metode_terbaik = st.session_state["best_method"]
    k_terbaik = st.session_state["best_k"]

    # Generate label berdasarkan metode terbaik
    labels_final = None
    if "AHC" in metode_terbaik:
        model_final = AgglomerativeClustering(n_clusters=k_terbaik, linkage='ward')
        labels_final = model_final.fit_predict(X_norm)
    elif "FCM" in metode_terbaik:
        data_final_T = X_norm.T
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_final_T, c=k_terbaik, m=2, error=0.005, maxiter=1000, init=None
        )
        labels_final = np.argmax(u, axis=0)

    # Ubah cluster dari 0-based → 1-based
    labels_final = labels_final + 1

    # Simpan ke DataFrame baru
    df_hasil = df.copy()
    df_hasil["Cluster"] = labels_final

    # Urutkan berdasarkan Cluster
    df_hasil = df_hasil.sort_values(by="Cluster").reset_index(drop=True)

    # Tampilkan preview
    st.dataframe(df_hasil.head(15))

    # Simpan ke CSV
    csv_hasil = df_hasil.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Hasil Clustering (CSV)",
        data=csv_hasil,
        file_name="hasil_clustering.csv",
        mime="text/csv"
    )


if selected == "Implementation":
    st.markdown("<h1 style='text-align: center;'>IMPLEMENTASI</h1>", unsafe_allow_html=True)

    if "best_method" not in st.session_state:
        st.warning("⚠️ Silakan jalankan dulu menu *Clustering* untuk menentukan metode terbaik.")
    else:
        best_method = st.session_state["best_method"]
        best_k = st.session_state["best_k"]

        st.info(f"📌 Prediksi menggunakan metode terbaik hasil clustering sebelumnya: "
                f"**{best_method}** (K={best_k})")

        # === Form input ===
        with st.form("form_batik"):
            nama_usaha = st.text_input("Nama Usaha")
            alamat = st.text_area("Alamat")
            lama_usaha = st.number_input("Lama Usaha (tahun)", min_value=0)
            kemitraan = st.number_input("Kemitraan", min_value=0)
            aset = st.number_input("Aset (jutaan)", min_value=0)
            omzet = st.number_input("Omzet (ribuan) per bulan", min_value=0)
            jml_naker = st.number_input("Jumlah Tenaga Kerja", min_value=0)
            izin = st.selectbox("Surat Izin", ("tidak memiliki", "proses pengurusan", "SIUP"))
            submit_pred = st.form_submit_button("Prediksi Cluster")

        if submit_pred:
            # --- Mapping surat izin ---
            izin_map = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}

            input_data = pd.DataFrame([{
                "lama usaha": lama_usaha,
                "kemitraan": kemitraan,
                "aset (jutaan)": aset,
                "omzet (ribuan) perbulan": omzet,
                "jml_naker": jml_naker,
                "surat Izin": izin_map[izin]
            }])

            # --- Load & preprocessing dataset asli ---
            df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
            df_new = df.drop(["nama_usaha", "alamat", "tahun"], axis=1)

            num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
            df_new["surat Izin"] = df_new["surat Izin"].map(num)

            # Normalisasi
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(df_new)

            input_data = input_data[df_new.columns]
            input_scaled = scaler.transform(input_data)

            # ==============================
            # Training model hanya sekali
            # ==============================
            if "model_fitted" not in st.session_state:
                if "AHC" in best_method:
                    model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
                    labels = model.fit_predict(X_scaled)

                    # Simpan centroid manual (mean tiap cluster)
                    centroids = []
                    for k in range(best_k):
                        centroids.append(X_scaled[labels == k].mean(axis=0))
                    centroids = np.array(centroids)

                    st.session_state.model_fitted = {"labels": labels, "centroids": centroids}

                elif "FCM" in best_method:
                    data_T = X_scaled.T
                    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                        data_T, c=best_k, m=2, error=0.005, maxiter=1000, init=None, seed=42
                    )
                    st.session_state.model_fitted = {"cntr": cntr}

            # ==============================
            # Prediksi cluster
            # ==============================
            cluster_label = None
            if "AHC" in best_method:
                centroids = st.session_state.model_fitted["centroids"]
                # cari centroid terdekat
                dists = np.linalg.norm(centroids - input_scaled, axis=1)
                cluster_label = np.argmin(dists) + 1

            elif "FCM" in best_method:
                cntr = st.session_state.model_fitted["cntr"]
                u_pred, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
                    input_scaled.T, cntr, m=2, error=0.005, maxiter=1000
                )
                cluster_label = np.argmax(u_pred, axis=0)[0] + 1

            # --- Output hasil prediksi ---
            st.subheader("📊 Hasil Prediksi Cluster")
            st.success(
                f"UMKM **{nama_usaha}** masuk ke **Cluster {cluster_label}** "
                f"(Metode: {best_method}, K={best_k})"
            )

            # ==============================
            # Simpan hasil ke DataFrame & CSV
            # ==============================
            hasil_prediksi = pd.DataFrame([{
                "nama_usaha": nama_usaha,
                "alamat": alamat,
                "lama usaha": lama_usaha,
                "kemitraan": kemitraan,
                "aset (jutaan)": aset,
                "omzet (ribuan) perbulan": omzet,
                "jml_naker": jml_naker,
                "surat Izin": izin,
                "Cluster": cluster_label,
                "Metode": best_method,
                "K": best_k
            }])

            csv = hasil_prediksi.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv,
                file_name="hasil_prediksi_cluster.csv",
                mime="text/csv"
            )





    st.write("---")
    st.write("By Fahrurrohman Ibnu Irsad Argyanto")
    st.write("© Copyright 2025.")

