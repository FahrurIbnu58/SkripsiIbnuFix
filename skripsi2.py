import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# -------------------------
# Konfigurasi halaman
# -------------------------
st.set_page_config(page_title='PERBANDINGAN METODE AHC DAN FCM UNTUK KLASTERISASI DATASET UMUM')
st.write("---")
st.markdown("<h1 style='text-align: center;'>TUGAS AKHIR</h1>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# Menu utama
# -------------------------
selected = option_menu(
    menu_title=None,
    options=["Description", "Preprocessing", "Entropy Weighting", "Clustering"],
    icons=["house","calculator","clipboard", "table"],
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
# Fungsi load dataset
# -------------------------
def get_dataset():
    if "df" in st.session_state:
        return st.session_state["df"].copy()
    else:
        st.warning("⚠️ Silakan upload dataset di menu *Description* terlebih dahulu.")
        return None

# -------------------------
# Description
# -------------------------
if selected == "Description":
    st.markdown("<h1 style='text-align: center;'>DESKRIPSI DATASET</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload file CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("✅ Dataset berhasil diupload dan siap diproses.")
        st.dataframe(df.head(20))
        st.write(f"- Jumlah Baris: {df.shape[0]}")
        st.write(f"- Jumlah Kolom: {df.shape[1]}")
    else:
        st.info("Silakan upload dataset CSV terlebih dahulu.")

# -------------------------
# Preprocessing
# -------------------------
if selected == "Preprocessing":
    st.markdown("<h1 style='text-align: center;'>PREPROCESSING DATA</h1>", unsafe_allow_html=True)

    df = get_dataset()
    if df is not None:
        st.subheader("Dataset Asli")
        st.dataframe(df)

        # --- Hapus fitur tidak relevan sesuai pilihan user ---
        fitur_hapus = st.multiselect("Pilih fitur yang ingin dihapus:", df.columns.tolist())
        df_new = df.drop(columns=fitur_hapus)
        st.subheader("Dataset Setelah Menghapus Fitur")
        st.dataframe(df_new)

        # --- Pilih fitur kategorikal untuk label encoding ---
        fitur_kategorikal = st.multiselect("Pilih fitur kategorikal untuk di-label encoding:", df_new.columns.tolist())
        df_encoded = df_new.copy()

        for col in fitur_kategorikal:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

        st.subheader("Dataset Setelah Label Encoding")
        st.dataframe(df_encoded)

        # --- Normalisasi hanya kolom numerik ---
        df_numeric = df_encoded.select_dtypes(include=[np.number])
        scaler_norm = MinMaxScaler()
        X_norm = scaler_norm.fit_transform(df_numeric)
        df_scaled = pd.DataFrame(X_norm, columns=df_numeric.columns)

        # Gabungkan hasil normalisasi
        df_compare = pd.concat(
            [df_encoded.reset_index(drop=True), df_scaled],
            axis=1,
            keys=["Data Asli", "Data Normalisasi"]
        )

        st.subheader("Perbandingan Data Asli vs Data Normalisasi")
        st.dataframe(df_compare)

        # --- Tombol Simpan ---
        if st.button("💾 Simpan Data Hasil Preprocessing"):
            st.session_state["df_new"] = df_encoded
            st.session_state["df_scaled"] = df_scaled
            st.success("✅ Data berhasil disimpan, silakan lanjut ke menu Entropy Weighting.")

# -------------------------
# Entropy Weighting
# -------------------------
if selected == "Entropy Weighting":
    st.markdown("<h1 style='text-align: center;'>ENTROPY WEIGHTING</h1>", unsafe_allow_html=True)

    df_scaled = st.session_state.get("df_scaled")

    if df_scaled is not None and not df_scaled.empty:
        weights_entropy = entropy_weighting(df_scaled.values)
        df_entropy_weighted = df_scaled * weights_entropy

        st.subheader("📌 Bobot Entropy untuk Setiap Fitur")
        weight_df = pd.DataFrame({
            "Fitur": df_scaled.columns,
            "Bobot": weights_entropy
        }).sort_values(by="Bobot", ascending=False).reset_index(drop=True)
        st.dataframe(weight_df)

        st.subheader("📊 Data Hasil Pembobotan Entropy")
        st.dataframe(df_entropy_weighted)

        # Simpan ke session_state
        st.session_state["weights_entropy"] = weights_entropy
        st.session_state["df_entropy_weighted"] = df_entropy_weighted
        st.session_state["weight_df"] = weight_df
        st.success("✅ Hasil entropy weighting disimpan, silakan lanjut ke menu Clustering.")
    else:
        st.warning("⚠️ Data belum tersedia. Lakukan preprocessing dan simpan data terlebih dahulu.")


# -------------------------
# Clustering
# -------------------------
if selected == "Clustering":
    st.markdown("<h1 style='text-align: center;'>CLUSTERING</h1>", unsafe_allow_html=True)

    df_scaled = st.session_state.get("df_scaled")
    df_entropy_weighted = st.session_state.get("df_entropy_weighted")

    if df_scaled is not None and df_entropy_weighted is not None:
        scenarios = {
            "AHC tanpa pembobotan": df_scaled,
            "FCM tanpa pembobotan": df_scaled,
            "AHC dengan pembobotan": df_entropy_weighted,
            "FCM dengan pembobotan": df_entropy_weighted,
        }

        results = {}

        for name, data in scenarios.items():
            scores = []
            for k in range(2, 11):
                if "AHC" in name:
                    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                    labels = model.fit_predict(data)
                else:
                    data_T = data.T
                    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                        data_T, c=k, m=2, error=0.005, maxiter=1000
                    )
                    labels = np.argmax(u, axis=0)

                score = silhouette_score(data, labels)
                scores.append((k, score))

            # cari nilai terbaik
            best_k, best_score = max(scores, key=lambda x: x[1])
            results[name] = {
                "scores": scores,
                "best_k": best_k,
                "best_score": best_score,
            }

            # tampilkan teks hasil per skenario
            st.write(f"### 🔹 {name}")
            for k, s in scores:
                st.write(f"- k = {k} → Silhouette = `{s:.4f}`")
            st.info(f"👉 Nilai terbaik: `{best_score:.4f}` pada k = `{best_k}`")

            # tampilkan diagram batang per skenario (dengan nilai di atas batang)
            fig, ax = plt.subplots()
            ks = [s[0] for s in scores]
            vals = [s[1] for s in scores]
            bars = ax.bar(ks, vals, color="skyblue", alpha=0.8)

            for bar, val, k in zip(bars, vals, ks):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}\nk={k}", 
                        ha="center", va="bottom", fontsize=8)

            ax.set_title(f"Nilai Silhouette - {name}")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Silhouette Coefficient")
            st.pyplot(fig)

        # === Bandingkan semua skenario ===
        best_scenario = max(results.items(), key=lambda x: x[1]["best_score"])

        st.subheader("📊 Perbandingan Silhouette Optimal dari Semua Skenario")
        compare_df = pd.DataFrame({
            "Skenario": list(results.keys()),
            "Silhouette Optimal": [v["best_score"] for v in results.values()],
            "k Optimal": [v["best_k"] for v in results.values()],
        }).sort_values(by="Silhouette Optimal", ascending=False).reset_index(drop=True)
        st.dataframe(compare_df)

        # Visualisasi gabungan
        fig2, ax2 = plt.subplots()
        bars = ax2.bar(compare_df["Skenario"], compare_df["Silhouette Optimal"], color="orange", alpha=0.8)
        for bar, val, k in zip(bars, compare_df["Silhouette Optimal"], compare_df["k Optimal"]):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}\nk={k}", 
                     ha="center", va="bottom", fontsize=8)

        ax2.set_title("Perbandingan Silhouette Optimal Antar Skenario")
        ax2.set_ylabel("Silhouette Coefficient")
        plt.xticks(rotation=20)
        st.pyplot(fig2)

        # tampilkan teks skenario terbaik
        st.success(
            f"🎯 Skenario terbaik adalah **{best_scenario[0]}** "
            f"dengan nilai silhouette = `{best_scenario[1]['best_score']:.4f}` "
            f"pada k = `{best_scenario[1]['best_k']}`"
        )

        # Simpan hasil clustering terbaik ke CSV
        st.download_button(
            "⬇️ Download Hasil Clustering Terbaik (CSV)",
            compare_df.to_csv(index=False).encode("utf-8"),
            "hasil_clustering.csv",
            "text/csv",
        )
    else:
        st.warning("⚠️ Data belum tersedia. Lakukan preprocessing dan entropy weighting terlebih dahulu.")




