import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import skfuzzy as fuzz
from streamlit_option_menu import option_menu

# ======================================================
# Guard: pastikan dependency Excel tersedia
# ======================================================
try:
    import openpyxl  # noqa: F401
except Exception:
    st.error("Library 'openpyxl' belum terinstall. Install dulu dengan:  pip install openpyxl")
    st.stop()

# -------------------------------------
# 🌈 Styling Page
# -------------------------------------
st.set_page_config(
    page_title='PERBANDINGAN METODE AHC DAN FCM UNTUK KLASTERISASI',
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
        .small-note { color:#546e7a; font-size:0.9rem; }
        .hint { color:#455a64; font-size:0.9rem; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------
# Judul utama halaman
# -------------------------------------
st.markdown("""
<h1 style='text-align: center; font-size: 30px;'>
Perbandingan Metode Agglomerative Hierarchical Clustering (AHC) dan Fuzzy C-Means (FCM) Untuk Klasterisasi (Input Excel)
</h1>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ======================================================
# Helper functions
# ======================================================
def entropy_weighting(data: np.ndarray) -> np.ndarray:
    """Entropy weighting (expects numeric, non-null)."""
    data = np.array(data, dtype=float)
    m, n = data.shape
    data = data - np.min(data, axis=0) + 1e-12
    col_sum = data.sum(axis=0) + 1e-12
    P = data / col_sum
    k = 1.0 / np.log(m)
    entropy = -k * (P * np.log(P + 1e-12)).sum(axis=0)
    weights = (1 - entropy) / (n - entropy.sum() + 1e-12)
    return weights

def safe_corr(a, b) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) < 3:
        return 0.0
    if np.isclose(np.std(a), 0.0) or np.isclose(np.std(b), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def plot_metric_curve(range_n_clusters, values, best_k, title, ylabel):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(list(range_n_clusters), values, marker='o')
    ax.set_title(title, pad=12)
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel(ylabel)
    ax.axvline(best_k, linestyle='--', linewidth=1.5)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig

def ensure_session_defaults():
    st.session_state.setdefault("df_raw", None)
    st.session_state.setdefault("df_preprocessed", None)
    st.session_state.setdefault("df_scaled", None)
    st.session_state.setdefault("scaler", None)
    st.session_state.setdefault("encoders", {})
    st.session_state.setdefault("drop_cols", [])
    st.session_state.setdefault("cat_cols", [])
    st.session_state.setdefault("status_preprocess_ok", False)
    st.session_state.setdefault("sheet_name", None)

ensure_session_defaults()

# ======================================================
# Menu Navigasi (Implementation DIHAPUS)
# ======================================================
selected = option_menu(
    menu_title=None,
    options=["Description", "Preprocessing", "Entropy Weighting", "Clustering"],
    icons=["info-circle", "tools", "activity", "diagram-3"],
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
# Sidebar: Upload Data Excel
# ======================================================
with st.sidebar:
    st.markdown("### 📤 Upload Dataset (Excel)")
    uploaded = st.file_uploader("Upload file Excel (.xlsx / .xls)", type=["xlsx", "xls"])

    if uploaded is not None:
        try:
            xls = pd.ExcelFile(uploaded)
            sheet_names = xls.sheet_names

            sheet = st.selectbox("Pilih sheet", sheet_names, index=0)
            df = pd.read_excel(xls, sheet_name=sheet)

            st.session_state["df_raw"] = df
            st.session_state["sheet_name"] = sheet
            st.success("✅ Dataset Excel berhasil di-load.")
            st.caption(f"Sheet: {sheet}")
            st.caption(f"Baris: {df.shape[0]} | Kolom: {df.shape[1]}")

        except Exception as e:
            st.session_state["df_raw"] = None
            st.error(f"Gagal membaca Excel: {e}")

    st.markdown("---")
    st.markdown("### ♻️ Reset")
    if st.button("Reset semua proses"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ======================================================
# Description
# ======================================================
if selected == "Description":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>📘 DESKRIPSI DATASET</h2>", unsafe_allow_html=True)

        if st.session_state["df_raw"] is None:
            st.warning("⚠️ Silakan upload dataset Excel di sidebar dulu.")
            st.markdown("<p class='small-note'>Sekarang input file hanya Excel (.xlsx/.xls).</p>", unsafe_allow_html=True)
        else:
            df = st.session_state["df_raw"]
            st.dataframe(df, use_container_width=True)
            st.caption(f"📊 Dataset berisi {df.shape[0]} baris dan {df.shape[1]} kolom.")
            st.success("✅ Dataset siap diproses di menu Preprocessing.")

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Preprocessing
# ======================================================
if selected == "Preprocessing":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>⚙️ PREPROCESSING DATA (DINAMIS)</h2>", unsafe_allow_html=True)

        if st.session_state["df_raw"] is None:
            st.warning("⚠️ Silakan upload dataset Excel di sidebar dulu.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            df = st.session_state["df_raw"].copy()
            st.write("### 🔍 Data Asli (Preview)")
            st.dataframe(df.head(20), use_container_width=True)

            st.markdown("---")
            st.write("### 1) Menghapus Kolom Tidak Relevan (Opsional)")
            all_cols = df.columns.tolist()
            drop_cols = st.multiselect(
                "Pilih kolom yang ingin dihapus",
                options=all_cols,
                default=st.session_state.get("drop_cols", [])
            )

            st.write("### 2) Penanganan Missing Value")
            missing_strategy = st.selectbox(
                "Pilih strategi untuk nilai kosong (NaN)",
                options=[
                    "Biarkan (akan error jika ada NaN saat proses numeric)",
                    "Hapus baris yang ada NaN",
                    "Isi NaN dengan 0 (untuk numerik)",
                    "Isi NaN dengan mean (untuk numerik)",
                    "Isi NaN dengan median (untuk numerik)",
                    "Isi NaN dengan modus (untuk kategorikal & numerik)",
                ],
                index=1
            )

            st.write("### 3) Label Encoding (Opsional)")
            has_categorical = st.radio(
                "Apakah ada kolom kategorikal (teks/kategori)?",
                ["Tidak", "Ya"],
                horizontal=True
            )

            cat_cols = []
            if has_categorical == "Ya":
                suggested = [c for c in df.columns if str(df[c].dtype) in ["object", "category"]]
                cat_cols = st.multiselect(
                    "Pilih kolom yang ingin di-encode menjadi numerik",
                    options=df.columns.tolist(),
                    default=st.session_state.get("cat_cols", suggested)
                )
                st.markdown("<div class='hint'>Catatan: LabelEncoder akan memberi angka 0..n untuk setiap kategori.</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.write("### 4) Jalankan Preprocessing")
            st.caption("Output preprocessing akan disimpan untuk dipakai di Entropy Weighting & Clustering.")

            if st.button("▶️ Jalankan Preprocessing"):
                work = df.copy()

                # Drop columns
                if drop_cols:
                    work = work.drop(columns=drop_cols, errors="ignore")

                # Missing handling
                if missing_strategy == "Hapus baris yang ada NaN":
                    work = work.dropna(axis=0)
                elif missing_strategy == "Isi NaN dengan 0 (untuk numerik)":
                    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
                    work[num_cols] = work[num_cols].fillna(0)
                elif missing_strategy == "Isi NaN dengan mean (untuk numerik)":
                    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
                    for c in num_cols:
                        work[c] = work[c].fillna(work[c].mean())
                elif missing_strategy == "Isi NaN dengan median (untuk numerik)":
                    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
                    for c in num_cols:
                        work[c] = work[c].fillna(work[c].median())
                elif missing_strategy == "Isi NaN dengan modus (untuk kategorikal & numerik)":
                    for c in work.columns:
                        if work[c].isna().any():
                            try:
                                mode_val = work[c].mode(dropna=True).iloc[0]
                                work[c] = work[c].fillna(mode_val)
                            except Exception:
                                pass

                # Encoding
                encoders = {}
                if has_categorical == "Ya" and cat_cols:
                    for c in cat_cols:
                        if c in work.columns:
                            le = LabelEncoder()
                            work[c] = le.fit_transform(work[c].astype(str))
                            encoders[c] = le

                numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) == 0:
                    st.error("❌ Tidak ada kolom numerik untuk diproses. Pilih kolom kategorikal untuk encoding atau pastikan ada kolom numerik.")
                    st.stop()

                st.session_state["df_preprocessed"] = work
                st.session_state["drop_cols"] = drop_cols
                st.session_state["cat_cols"] = cat_cols
                st.session_state["encoders"] = encoders

                scaler = MinMaxScaler()
                X_norm = scaler.fit_transform(work[numeric_cols])
                df_scaled = pd.DataFrame(X_norm, columns=numeric_cols)

                st.session_state["df_scaled"] = df_scaled
                st.session_state["scaler"] = scaler
                st.session_state["status_preprocess_ok"] = True

                st.success("✅ Preprocessing selesai. Hasil ditampilkan di bawah.")

            if st.session_state.get("status_preprocess_ok", False):
                st.markdown("---")
                st.subheader("✅ Hasil Setelah Drop/Encoding/Missing Handling")
                st.dataframe(st.session_state["df_preprocessed"].head(50), use_container_width=True)

                st.subheader("✅ Hasil Normalisasi (0–1)")
                st.dataframe(st.session_state["df_scaled"].head(50), use_container_width=True)

                st.markdown("### 📥 Download Hasil")
                c1, c2 = st.columns(2)
                with c1:
                    csv_pre = st.session_state["df_preprocessed"].to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Preprocessed CSV", csv_pre, "preprocessed.csv", "text/csv")
                with c2:
                    csv_scaled = st.session_state["df_scaled"].to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Normalized CSV", csv_scaled, "normalized.csv", "text/csv")

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Entropy Weighting
# ======================================================
if selected == "Entropy Weighting":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>📊 ENTROPY WEIGHTING (DINAMIS)</h2>", unsafe_allow_html=True)

        if not st.session_state.get("status_preprocess_ok", False):
            st.warning("⚠️ Jalankan menu *Preprocessing* dulu sampai normalisasi selesai.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            df_scaled = st.session_state["df_scaled"].copy()

            st.write("### Data Normalisasi (Input Entropy)")
            st.dataframe(df_scaled.head(50), use_container_width=True)

            weights_entropy = entropy_weighting(df_scaled.values)
            df_entropy_weighted = df_scaled * weights_entropy

            weight_df = (
                pd.DataFrame({"Fitur": df_scaled.columns, "Bobot": weights_entropy})
                .sort_values(by="Bobot", ascending=False)
                .reset_index(drop=True)
            )

            st.subheader("🔎 Bobot Entropy per Fitur")
            st.dataframe(weight_df, use_container_width=True)

            st.subheader("✅ Data Setelah Entropy Weighting")
            st.dataframe(df_entropy_weighted.head(50), use_container_width=True)

            st.session_state["weights_entropy"] = weights_entropy
            st.session_state["df_entropy_weighted"] = df_entropy_weighted
            st.session_state["feature_ranking"] = weight_df

            csv_w = df_entropy_weighted.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Entropy Weighted CSV", csv_w, "entropy_weighted.csv", "text/csv")

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Clustering
# ======================================================
if selected == "Clustering":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>🔗 CLUSTERING: AHC vs FCM (DINAMIS)</h2>", unsafe_allow_html=True)

        if not st.session_state.get("status_preprocess_ok", False):
            st.warning("⚠️ Jalankan menu *Preprocessing* dulu sampai normalisasi selesai.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            df_scaled = st.session_state["df_scaled"].copy()
            X_norm = df_scaled.values

            st.write("### Pengaturan")
            range_min, range_max = st.slider("Rentang jumlah cluster (k)", 2, 20, (2, 10))
            range_n_clusters = range(range_min, range_max + 1)

            use_entropy = st.checkbox("Gunakan seleksi fitur dari Entropy Weighting?", value=True)

            if use_entropy and "df_entropy_weighted" not in st.session_state:
                weights_entropy = entropy_weighting(df_scaled.values)
                df_entropy_weighted = df_scaled * weights_entropy
                feature_ranking = (
                    pd.DataFrame({"Feature": df_scaled.columns, "Weight": weights_entropy})
                    .sort_values(by="Weight", ascending=False)
                    .reset_index(drop=True)
                )
                st.session_state["df_entropy_weighted"] = df_entropy_weighted
                st.session_state["feature_ranking"] = feature_ranking

            if use_entropy:
                feature_ranking = st.session_state["feature_ranking"]
                top_n = st.slider("Ambil Top-N fitur terbaik (Entropy)", 2, min(20, df_scaled.shape[1]), min(5, df_scaled.shape[1]))
                selected_features = feature_ranking["Feature"].iloc[:top_n].tolist()
                X_sub = st.session_state["df_entropy_weighted"][selected_features].values
            else:
                selected_features = df_scaled.columns.tolist()
                X_sub = X_norm

            st.caption(f"Fitur yang dipakai ({len(selected_features)}): {', '.join(selected_features)}")

            def eval_scenario(name, X, is_fcm=False):
                sil_list, dbi_list, ch_list = [], [], []
                for k in range_n_clusters:
                    if not is_fcm:
                        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
                    else:
                        _, u, *_ = fuzz.cluster.cmeans(
                            X.T, c=k, m=2, error=0.005, maxiter=1000, init=None, seed=42
                        )
                        labels = np.argmax(u, axis=0)

                    if len(np.unique(labels)) < 2:
                        sil, dbi, ch = -1, np.inf, -1
                    else:
                        sil = silhouette_score(X, labels)
                        dbi = davies_bouldin_score(X, labels)
                        ch = calinski_harabasz_score(X, labels)

                    sil_list.append(float(sil))
                    dbi_list.append(float(dbi))
                    ch_list.append(float(ch))

                return {"Metode": name, "X": X, "is_fcm": is_fcm, "sil": sil_list, "dbi": dbi_list, "ch": ch_list}

            st.markdown("---")
            if st.button("▶️ Jalankan Evaluasi Clustering"):
                scenario_results = [
                    eval_scenario("AHC", X_sub, is_fcm=False),
                    eval_scenario("FCM", X_sub, is_fcm=True),
                ]

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

                st.subheader("📈 Kurva Evaluasi per Metode")
                recap_rows = []
                ks = list(range_n_clusters)

                for s in scenario_results:
                    name = s["Metode"]
                    sil = s["sil"]
                    dbi = s["dbi"]

                    best_k_sil = ks[int(np.argmax(sil))]
                    best_sil = float(np.max(sil))

                    best_k_dbi = ks[int(np.argmin(dbi))]
                    best_dbi = float(np.min(dbi))

                    c1, c2 = st.columns(2)
                    with c1:
                        st.pyplot(plot_metric_curve(range_n_clusters, sil, best_k_sil, f"{name} - Silhouette", "Silhouette"))
                    with c2:
                        st.pyplot(plot_metric_curve(range_n_clusters, dbi, best_k_dbi, f"{name} - DBI", "DBI"))

                    recap_rows.append({
                        "Metode": name,
                        "BestK_Silhouette": best_k_sil,
                        "Silhouette_Max": best_sil,
                        "BestK_DBI": best_k_dbi,
                        "DBI_Min": best_dbi
                    })

                df_recap = pd.DataFrame(recap_rows)
                st.dataframe(df_recap, use_container_width=True)

                if chosen_metric == "Silhouette":
                    best_row = df_recap.loc[df_recap["Silhouette_Max"].idxmax()]
                    best_method = best_row["Metode"]
                    best_k = int(best_row["BestK_Silhouette"])
                    best_value = float(best_row["Silhouette_Max"])
                    st.success(f"✅ Metode terbaik: **{best_method}** | K={best_k} | Silhouette={best_value:.4f}")
                else:
                    best_row = df_recap.loc[df_recap["DBI_Min"].idxmin()]
                    best_method = best_row["Metode"]
                    best_k = int(best_row["BestK_DBI"])
                    best_value = float(best_row["DBI_Min"])
                    st.success(f"✅ Metode terbaik: **{best_method}** | K={best_k} | DBI={best_value:.4f}")

                st.markdown("---")
                st.subheader("📌 Hasil Clustering (Final)")

                if best_method == "AHC":
                    labels_best = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X_sub)
                else:
                    _, u_best, *_ = fuzz.cluster.cmeans(
                        X_sub.T, c=best_k, m=2, error=0.005, maxiter=1000, init=None, seed=42
                    )
                    labels_best = np.argmax(u_best, axis=0)

                df_out = st.session_state["df_preprocessed"].copy()
                df_out["Cluster"] = labels_best + 1

                st.dataframe(df_out, use_container_width=True)

                st.write("📌 **Jumlah anggota tiap cluster**")
                st.dataframe(
                    df_out["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Jumlah Data"),
                    use_container_width=True
                )

                csv_hasil = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Hasil Clustering (CSV)",
                    data=csv_hasil,
                    file_name="hasil_clustering.csv",
                    mime="text/csv"
                )

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Footer
# ======================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    © 2025 — Klasterisasi (By Fahrurrohman Ibnu Irsad Argyanto)
</div>
""", unsafe_allow_html=True)
