import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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

def safe_corr(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) < 3:
        return 0.0
    if np.isclose(np.std(a), 0.0) or np.isclose(np.std(b), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

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

        df_new = df.drop(['nama_usaha', 'alamat', 'tahun'], axis=1)
        st.subheader("Menghapus Kolom yang Tidak Relevan")
        st.dataframe(df_new)

        st.subheader("Label Encoding Kolom 'surat Izin'")
        num = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
        df_new['surat Izin'] = df_new['surat Izin'].map(num)
        st.dataframe(df_new)

        st.subheader("Normalisasi Data (0–1)")
        scaler_norm = MinMaxScaler()
        X_norm = scaler_norm.fit_transform(df_new)
        df_scaled = pd.DataFrame(X_norm, columns=df_new.columns)
        st.dataframe(df_scaled)

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

        weight_df = pd.DataFrame({"Fitur": df_new.columns, "Bobot": weights_entropy}).sort_values(by="Bobot", ascending=False)
        st.dataframe(weight_df.reset_index(drop=True))
        st.dataframe(df_entropy_weighted)

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Clustering Section
# ======================================================
if selected == "Clustering":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>🔗 CLUSTERING: AHC vs FCM</h2>", unsafe_allow_html=True)

        st.write("""Skenario Uji:
        1) AHC Tanpa Seleksi Fitur
        2) FCM Tanpa Seleksi Fitur
        3) AHC Dengan Seleksi Fitur (Entropy Weighting)
        4) FCM Dengan Seleksi Fitur (Entropy Weighting)
        """)

        # ===== Load & preprocess =====
        df = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
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

        # =====================================================
        # Evaluasi semua skenario (Silhouette, DBI, CH)
        # =====================================================
        scenario_results = []

        def eval_scenario(name, X, is_fcm=False):
            sil_list, dbi_list, ch_list = [], [], []
            labels_per_k = {}

            for k in range_n_clusters:
                if not is_fcm:
                    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
                else:
                    cntr, u, *_ = fuzz.cluster.cmeans(
                        X.T, c=k, m=2, error=0.005, maxiter=1000, init=None, seed=42
                    )
                    labels = np.argmax(u, axis=0)

                sil_list.append(silhouette_score(X, labels))
                dbi_list.append(davies_bouldin_score(X, labels))
                ch_list.append(calinski_harabasz_score(X, labels))

                labels_per_k[k] = labels

            return {
                "Skenario": name,
                "X": X,
                "is_fcm": is_fcm,
                "sil": sil_list,
                "dbi": dbi_list,
                "ch": ch_list,
                "labels_per_k": labels_per_k
            }

        scenario_results.append(eval_scenario("AHC Tanpa Seleksi Fitur", X_norm, is_fcm=False))
        scenario_results.append(eval_scenario("FCM Tanpa Seleksi Fitur", X_norm, is_fcm=True))
        scenario_results.append(eval_scenario("AHC Dengan Seleksi Fitur", X_sub, is_fcm=False))
        scenario_results.append(eval_scenario("FCM Dengan Seleksi Fitur", X_sub, is_fcm=True))

        # =====================================================
        # VALIDASI tunggal (CH) untuk memilih metrik evaluasi (Silhouette vs DBI)
        # =====================================================
        all_ch = np.concatenate([np.array(s["ch"]) for s in scenario_results])
        all_sil = np.concatenate([np.array(s["sil"]) for s in scenario_results])
        all_dbi = np.concatenate([np.array(s["dbi"]) for s in scenario_results])

        corr_ch_sil = safe_corr(all_ch, all_sil)
        corr_ch_dbi_inv = safe_corr(all_ch, -all_dbi)

        if corr_ch_sil >= corr_ch_dbi_inv:
            chosen_metric = "Silhouette"
            metric_rule = "maksimum"
        else:
            chosen_metric = "DBI"
            metric_rule = "minimum"

        st.subheader("✅ Validasi Tunggal (Calinski–Harabasz) → Pilih Metrik Evaluasi")
        st.write(f"- Korelasi CH vs Silhouette: **{corr_ch_sil:.4f}**")
        st.write(f"- Korelasi CH vs (-DBI): **{corr_ch_dbi_inv:.4f}**")
        st.success(f"➡️ Metrik evaluasi yang dipakai: **{chosen_metric}** (ambil nilai {metric_rule})")

        # =====================================================
        # Kurva per skenario + rekap (BestK Sil & BestK DBI)
        # =====================================================
        st.subheader("📈 Kurva Evaluasi per Skenario (Silhouette & DBI)")
        recap_rows = []

        for s in scenario_results:
            name = s["Skenario"]
            sil = s["sil"]
            dbi = s["dbi"]

            best_k_sil = list(range_n_clusters)[int(np.argmax(sil))]
            best_sil = float(np.max(sil))

            best_k_dbi = list(range_n_clusters)[int(np.argmin(dbi))]
            best_dbi = float(np.min(dbi))

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_metric_curve(range_n_clusters, sil, best_k_sil,
                                            f"{name} - Silhouette", "Silhouette", color="blue"))
            with c2:
                st.pyplot(plot_metric_curve(range_n_clusters, dbi, best_k_dbi,
                                            f"{name} - DBI", "DBI", color="black"))

            recap_rows.append({
                "Skenario": name,
                "BestK_Silhouette": best_k_sil,
                "Silhouette_Max": best_sil,
                "BestK_DBI": best_k_dbi,
                "DBI_Min": best_dbi
            })

        df_recap = pd.DataFrame(recap_rows)
        st.dataframe(df_recap, use_container_width=True)

        # =====================================================
        # Pilih metode terbaik berdasarkan metrik terpilih
        # =====================================================
        if chosen_metric == "Silhouette":
            best_row = df_recap.loc[df_recap["Silhouette_Max"].idxmax()]
            best_method = best_row["Skenario"]
            best_k = int(best_row["BestK_Silhouette"])
            best_value = float(best_row["Silhouette_Max"])
            st.success(f"✅ Metode terbaik berdasarkan Silhouette: **{best_method}** | K={best_k} | Silhouette={best_value:.4f}")
        else:
            best_row = df_recap.loc[df_recap["DBI_Min"].idxmin()]
            best_method = best_row["Skenario"]
            best_k = int(best_row["BestK_DBI"])
            best_value = float(best_row["DBI_Min"])
            st.success(f"✅ Metode terbaik berdasarkan DBI: **{best_method}** | K={best_k} | DBI={best_value:.4f}")

        st.session_state["best_method"] = best_method
        st.session_state["best_k"] = best_k
        st.session_state["best_metric"] = chosen_metric
        st.session_state["best_metric_value"] = best_value

        # =====================================================
        # REKAPITULASI hasil clustering metode terpilih (SETELAH metode terbaik)
        # =====================================================
        st.subheader("📌 Rekapitulasi Hasil Clustering (Metode Terpilih)")

        # Ambil skenario terpilih
        chosen_scenario = None
        for s in scenario_results:
            if s["Skenario"] == best_method:
                chosen_scenario = s
                break

        X_use = chosen_scenario["X"]
        is_fcm = chosen_scenario["is_fcm"]

        # tentukan data rekap berdasarkan skenario
        if "Dengan Seleksi Fitur" in best_method:
            df_recap_source = df[selected_features].copy()
        else:
            df_recap_source = df_new.copy()

        # label final untuk metode terbaik
        if not is_fcm:
            model_final_tmp = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
            labels_best = model_final_tmp.fit_predict(X_use)
        else:
            cntr_best, u_best, *_ = fuzz.cluster.cmeans(
                X_use.T, c=best_k, m=2, error=0.005, maxiter=1000, init=None, seed=42
            )
            labels_best = np.argmax(u_best, axis=0)

        # buat 1-based
        df_rekap = df_recap_source.copy()
        df_rekap["Cluster"] = labels_best + 1

        izin_col = "surat Izin" if "surat Izin" in df_rekap.columns else None

        # numerik untuk rentang min-max
        num_cols = [c for c in df_rekap.columns if c not in ["Cluster", izin_col] and pd.api.types.is_numeric_dtype(df_rekap[c])]

        # rentang min-max
        if num_cols:
            cluster_min = df_rekap.groupby("Cluster")[num_cols].min()
            cluster_max = df_rekap.groupby("Cluster")[num_cols].max()
            cluster_ranges = (cluster_min.astype(str) + " – " + cluster_max.astype(str)).reset_index()
        else:
            cluster_ranges = pd.DataFrame({"Cluster": sorted(df_rekap["Cluster"].unique())})

        # distribusi surat izin
        if izin_col:
            izin_dist = df_rekap.groupby(["Cluster", izin_col]).size().unstack(fill_value=0).reset_index()
            cluster_summary = pd.merge(cluster_ranges, izin_dist, on="Cluster", how="left")
        else:
            cluster_summary = cluster_ranges

        st.write("✅ **Ringkasan tiap cluster (rentang min–max + distribusi surat izin)**")
        st.dataframe(cluster_summary, use_container_width=True)

        st.write("📌 **Jumlah anggota tiap cluster**")
        st.dataframe(
            df_rekap["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Jumlah Data"),
            use_container_width=True
        )

        # simpan agar bisa dipakai lagi
        st.session_state["best_labels_1based"] = (labels_best + 1)
        st.session_state["cluster_summary_best"] = cluster_summary

        # =====================================================
        # Generate final labels & Download Hasil Clustering
        # =====================================================
        st.subheader("📥 Download Hasil Clustering")

        # Hasil clustering final dipakai untuk download: gunakan X_norm (biar konsisten dataset full),
        # tapi sesuai metode terpilih (tanpa/dengan seleksi fitur) kita gunakan X_use (sudah sesuai scenario).
        labels_final = labels_best  # sudah sesuai metode terbaik

        labels_final = np.array(labels_final) + 1  # 1-based

        df_hasil = df.copy()
        df_hasil["Cluster"] = labels_final
        df_hasil = df_hasil.sort_values(by="Cluster").reset_index(drop=True)
        st.session_state["df_clustered"] = df_hasil

        st.dataframe(df_hasil, use_container_width=True)

        csv_hasil = df_hasil.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Hasil Clustering (CSV)",
            data=csv_hasil,
            file_name="hasil_clustering.csv",
            mime="text/csv"
        )

        # simpan centroid untuk implementasi
        if "AHC" in best_method:
            # centroid mean (untuk prediksi jarak)
            centroids = np.array([X_use[labels_best == i].mean(axis=0) for i in range(best_k)])
            st.session_state["ahc_centroids"] = centroids
        else:
            st.session_state["fcm_cntr"] = cntr_best

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
            chosen_metric = st.session_state.get("best_metric", "-")
            metric_val = st.session_state.get("best_metric_value", "-")

            st.info(f"📌 Metode terbaik: **{best_method}** | K={best_k} | Metrik={chosen_metric} ({metric_val})")

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

                    df0 = pd.read_csv("DATA BATIK DINAS UMKM 1.csv")
                    df_new0 = df0.drop(["nama_usaha", "alamat", "tahun"], axis=1)
                    num0 = {"SIUP": 2, "proses pengurusan": 1, "tidak memiliki": 0}
                    df_new0["surat Izin"] = df_new0["surat Izin"].map(num0)

                    scaler0 = MinMaxScaler()
                    X_scaled0 = scaler0.fit_transform(df_new0)

                    input_data = input_data[df_new0.columns]
                    input_scaled = scaler0.transform(input_data)

                    cluster_label = None
                    if "AHC" in best_method:
                        if "ahc_centroids" not in st.session_state:
                            st.warning("Centroid AHC belum tersimpan. Jalankan Clustering dulu.")
                        else:
                            centroids = st.session_state["ahc_centroids"]
                            dists = np.linalg.norm(centroids - input_scaled, axis=1)
                            cluster_label = int(np.argmin(dists)) + 1

                    elif "FCM" in best_method:
                        if "fcm_cntr" not in st.session_state:
                            st.warning("Centroid FCM belum tersimpan. Jalankan Clustering dulu.")
                        else:
                            cntr = st.session_state["fcm_cntr"]
                            u_pred = fuzz.cluster.cmeans_predict(input_scaled.T, cntr, m=2, error=0.005, maxiter=1000)[0]
                            cluster_label = int(np.argmax(u_pred, axis=0)[0]) + 1

                    if cluster_label is not None:
                        st.subheader("📊 Hasil Prediksi Cluster")
                        st.success(f"UMKM **{nama_usaha}** masuk ke **Cluster {cluster_label}** (Metode: {best_method}, K={best_k})")

                        if "df_clustered" not in st.session_state:
                            st.session_state["df_clustered"] = df0.copy()

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
