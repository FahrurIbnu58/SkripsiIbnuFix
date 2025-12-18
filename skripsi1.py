import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    page_title="PERBANDINGAN METODE AHC DAN FCM UNTUK KLASTERISASI",
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
        .warn { color:#b71c1c; font-weight:600; }
        .ok { color:#1b5e20; font-weight:600; }
    </style>
""", unsafe_allow_html=True)

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

def plot_metric_curve(range_n_clusters, values, best_k, title, ylabel, color="blue"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(list(range_n_clusters), values, marker="o", color=color)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel(ylabel)
    ax.axvline(best_k, color="red", linestyle="--", linewidth=1.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig

def ensure_session_defaults():
    st.session_state.setdefault("df_raw", None)
    st.session_state.setdefault("sheet_name", None)

    st.session_state.setdefault("df_preprocessed", None)
    st.session_state.setdefault("df_scaled", None)
    st.session_state.setdefault("scaler", None)

    st.session_state.setdefault("drop_cols", [])
    st.session_state.setdefault("cat_cols", [])
    st.session_state.setdefault("encoders", {})          # boleh tetap ada (untuk inverse jika pakai LabelEncoder)
    st.session_state.setdefault("manual_maps", {})       # ✅ mapping manual per kolom
    st.session_state.setdefault("status_preprocess_ok", False)

    st.session_state.setdefault("weights_entropy", None)
    st.session_state.setdefault("df_entropy_weighted", None)
    st.session_state.setdefault("feature_ranking", None)
    st.session_state.setdefault("X_weighted", None)

    st.session_state.setdefault("best_method", None)
    st.session_state.setdefault("best_k", None)
    st.session_state.setdefault("best_metric", None)
    st.session_state.setdefault("best_metric_value", None)
    st.session_state.setdefault("df_clustered", None)
    st.session_state.setdefault("cluster_summary_best", None)

    st.session_state.setdefault("ahc_centroids", None)
    st.session_state.setdefault("fcm_cntr", None)

    st.session_state.setdefault("data_fingerprint", None)

ensure_session_defaults()

def fingerprint_df(df: pd.DataFrame) -> str:
    cols = tuple(map(str, df.columns.tolist()))
    return f"n={df.shape[0]}|m={df.shape[1]}|cols={cols}"

def sanitize_defaults(options: list, defaults: list) -> list:
    opt_set = set(options)
    return [d for d in (defaults or []) if d in opt_set]

def reset_downstream():
    st.session_state["df_preprocessed"] = None
    st.session_state["df_scaled"] = None
    st.session_state["scaler"] = None
    st.session_state["encoders"] = {}
    st.session_state["manual_maps"] = {}
    st.session_state["status_preprocess_ok"] = False

    st.session_state["weights_entropy"] = None
    st.session_state["df_entropy_weighted"] = None
    st.session_state["feature_ranking"] = None
    st.session_state["X_weighted"] = None

    st.session_state["best_method"] = None
    st.session_state["best_k"] = None
    st.session_state["best_metric"] = None
    st.session_state["best_metric_value"] = None
    st.session_state["df_clustered"] = None
    st.session_state["cluster_summary_best"] = None

    st.session_state["ahc_centroids"] = None
    st.session_state["fcm_cntr"] = None

def build_manual_mapping_ui(df: pd.DataFrame, cat_cols: list) -> tuple[dict, list]:
    """
    Return:
      - mapping_dict: {col: {label: code_int}}
      - errors: list of error strings (if any)
    """
    mapping_dict: dict[str, dict] = {}
    errors: list[str] = []

    if not cat_cols:
        return mapping_dict, errors

    st.markdown("### 3A) Manual Label Encoding (User Mapping)")
    st.caption("Atur angka untuk tiap label. Batas: 0 s/d (jumlah label - 1). Angka harus unik dalam satu kolom.")

    for col in cat_cols:
        # ambil label unik (string), urutkan biar stabil
        raw_vals = df[col].astype(str).fillna("NaN")
        labels = sorted(raw_vals.unique().tolist())
        n_labels = len(labels)

        if n_labels <= 1:
            st.info(f"Kolom **{col}** hanya punya 1 label unik, encoding tidak diperlukan.")
            continue

        st.markdown(f"#### Kolom: **{col}** (jumlah label: {n_labels})")
        st.markdown(f"<div class='hint'>Range angka yang boleh: 0 sampai {n_labels-1}</div>", unsafe_allow_html=True)

        # default mapping: pakai mapping lama kalau ada, kalau tidak 0..n-1 sesuai urutan labels
        old_map = st.session_state.get("manual_maps", {}).get(col, {})
        used_defaults = []
        for i, lab in enumerate(labels):
            if lab in old_map and isinstance(old_map[lab], (int, np.integer)):
                val = int(old_map[lab])
            else:
                val = i
            # pastikan masih dalam range
            if val < 0 or val > n_labels - 1:
                val = i
            used_defaults.append(val)

        col_map: dict[str, int] = {}
        picked_values = []

        # UI input angka per label
        for i, lab in enumerate(labels):
            key = f"map::{col}::{lab}"
            val = st.number_input(
                f"Nilai untuk label '{lab}'",
                min_value=0,
                max_value=n_labels - 1,
                value=int(used_defaults[i]),
                step=1,
                key=key
            )
            val = int(val)
            col_map[lab] = val
            picked_values.append(val)

        # validasi unik
        if len(set(picked_values)) != len(picked_values):
            errors.append(f"Kolom '{col}' punya angka duplikat. Setiap label harus punya angka unik 0..{n_labels-1}.")
            st.markdown("<div class='warn'>❌ Ada angka duplikat! Harus unik untuk tiap label.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ok'>✅ Mapping valid (unik).</div>", unsafe_allow_html=True)

        mapping_dict[col] = col_map
        st.markdown("---")

    return mapping_dict, errors

# ======================================================
# Menu Navigasi
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

            fp = fingerprint_df(df)
            if st.session_state["data_fingerprint"] != fp:
                st.session_state["data_fingerprint"] = fp
                st.session_state["drop_cols"] = []
                st.session_state["cat_cols"] = []
                reset_downstream()

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
            st.markdown("<p class='small-note'>Input file hanya Excel (.xlsx/.xls).</p>", unsafe_allow_html=True)
        else:
            df = st.session_state["df_raw"]
            st.dataframe(df, use_container_width=True)
            st.caption(f"📊 Dataset berisi {df.shape[0]} baris dan {df.shape[1]} kolom.")
            st.success("✅ Dataset siap diproses di menu Preprocessing.")

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Preprocessing (Manual Label Encoding)
# ======================================================
if selected == "Preprocessing":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>⚙️ PREPROCESSING DATA (DINAMIS)</h2>", unsafe_allow_html=True)

        if st.session_state["df_raw"] is None:
            st.warning("⚠️ Silakan upload dataset Excel di sidebar dulu.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        df = st.session_state["df_raw"].copy()
        st.write("### 🔍 Data Asli (Preview)")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("---")
        st.write("### 1) Menghapus Kolom Tidak Relevan (Opsional)")
        all_cols = df.columns.tolist()
        safe_default_drop = sanitize_defaults(all_cols, st.session_state.get("drop_cols", []))

        drop_cols = st.multiselect(
            "Pilih kolom yang ingin dihapus",
            options=all_cols,
            default=safe_default_drop
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

        st.write("### 3) Kolom Kategorikal yang Mau Di-Encode (Manual)")
        has_categorical = st.radio("Apakah ada kolom kategorikal (teks/kategori)?", ["Tidak", "Ya"], horizontal=True)

        # tentukan kandidat kolom kategorikal (object/category/bool)
        candidate_cat = [c for c in df.columns if str(df[c].dtype) in ["object", "category", "bool"]]
        cat_cols = []

        if has_categorical == "Ya":
            safe_default_cat = sanitize_defaults(df.columns.tolist(), st.session_state.get("cat_cols", candidate_cat))
            cat_cols = st.multiselect(
                "Pilih kolom yang ingin di-encode menjadi numerik (manual mapping)",
                options=df.columns.tolist(),
                default=safe_default_cat
            )
            st.markdown("<div class='hint'>Catatan: kamu yang menentukan angka untuk tiap label (0..n-1).</div>", unsafe_allow_html=True)

        st.markdown("---")

        # UI mapping manual (dibangun dari df yang masih asli)
        manual_maps, map_errors = build_manual_mapping_ui(df, cat_cols) if has_categorical == "Ya" else ({}, [])

        st.write("### 4) Jalankan Preprocessing")
        st.caption("Jika ada error mapping (duplikat angka), preprocessing akan dihentikan sampai valid.")

        if st.button("▶️ Jalankan Preprocessing"):
            if map_errors:
                st.error("❌ Mapping manual belum valid. Perbaiki dulu:")
                for err in map_errors:
                    st.write(f"- {err}")
                st.stop()

            work = df.copy()

            # drop columns
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

            # Manual Encoding
            # mapping dibuat dari df asli; pastikan kita encode kolom yang masih ada setelah drop
            used_maps = {}
            for c, cmap in manual_maps.items():
                if c in work.columns:
                    # map berdasarkan string
                    work[c] = work[c].astype(str).map(cmap)
                    used_maps[c] = cmap

            # Validasi: kalau ada nilai yang jadi NaN gara-gara label baru (misal ada NaN asli / label belum termap)
            # isi dengan -1 lalu warning (biar tidak crash)
            for c in used_maps.keys():
                if work[c].isna().any():
                    st.warning(f"⚠️ Kolom '{c}' ada label yang tidak termapping. Nilai tersebut akan diisi -1.")
                    work[c] = work[c].fillna(-1)

            # ambil kolom numerik untuk modeling
            numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                st.error("❌ Tidak ada kolom numerik untuk diproses. Pastikan ada kolom numerik atau lakukan encoding.")
                st.stop()

            st.session_state["df_preprocessed"] = work
            st.session_state["drop_cols"] = drop_cols
            st.session_state["cat_cols"] = cat_cols
            st.session_state["manual_maps"] = used_maps
            st.session_state["encoders"] = {}  # kosongkan karena kita pakai manual map

            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(work[numeric_cols])
            df_scaled = pd.DataFrame(X_norm, columns=numeric_cols)

            st.session_state["df_scaled"] = df_scaled
            st.session_state["scaler"] = scaler
            st.session_state["status_preprocess_ok"] = True

            # reset downstream
            st.session_state["weights_entropy"] = None
            st.session_state["df_entropy_weighted"] = None
            st.session_state["feature_ranking"] = None
            st.session_state["X_weighted"] = None

            st.session_state["best_method"] = None
            st.session_state["best_k"] = None
            st.session_state["best_metric"] = None
            st.session_state["best_metric_value"] = None
            st.session_state["df_clustered"] = None
            st.session_state["cluster_summary_best"] = None
            st.session_state["ahc_centroids"] = None
            st.session_state["fcm_cntr"] = None

            st.success("✅ Preprocessing selesai. Hasil ditampilkan di bawah.")

        if st.session_state.get("status_preprocess_ok", False):
            st.markdown("---")
            st.subheader("✅ Hasil Setelah Drop/Manual-Encoding/Missing Handling")
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
# Entropy Weighting (sesuai contoh + heatmap)
# ======================================================
if selected == "Entropy Weighting":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>📊 ENTROPY WEIGHTING (DINAMIS)</h2>", unsafe_allow_html=True)

        if not st.session_state.get("status_preprocess_ok", False):
            st.warning("⚠️ Jalankan menu *Preprocessing* dulu sampai normalisasi selesai.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        df_scaled = st.session_state["df_scaled"].copy()
        st.write("### Data Normalisasi (Input Entropy)")
        st.dataframe(df_scaled.head(50), use_container_width=True)

        weights_entropy = entropy_weighting(df_scaled.values)

        feature_ranking = pd.DataFrame({
            "Feature": df_scaled.columns,
            "Weight": weights_entropy
        }).sort_values(by="Weight", ascending=False).reset_index(drop=True)

        st.subheader("🔎 Ranking Fitur Berdasarkan Entropy Weighting")
        st.dataframe(feature_ranking, use_container_width=True)

        df_entropy_weighted = df_scaled * weights_entropy
        X_weighted = df_entropy_weighted.values

        st.subheader("✅ Data Setelah Entropy Weighting")
        st.dataframe(df_entropy_weighted.head(50), use_container_width=True)

        st.subheader("🔥 Heatmap Entropy Weighting - Feature Importance")
        heatmap_data = pd.DataFrame(
            [feature_ranking["Weight"].values],
            columns=feature_ranking["Feature"]
        )

        fig, ax = plt.subplots(figsize=(max(10, len(feature_ranking) * 0.8), 3))
        sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", cbar=False, fmt=".3f", ax=ax)
        ax.set_title("Entropy Weighting - Feature Importance", fontsize=14, weight="bold")
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)

        st.session_state["weights_entropy"] = weights_entropy
        st.session_state["feature_ranking"] = feature_ranking
        st.session_state["df_entropy_weighted"] = df_entropy_weighted
        st.session_state["X_weighted"] = X_weighted

        csv_weighted = df_entropy_weighted.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Entropy Weighted CSV",
            csv_weighted,
            "entropy_weighted.csv",
            "text/csv"
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Clustering (4 skenario)
# ======================================================
if selected == "Clustering":
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>🔗 CLUSTERING: AHC vs FCM (4 SKENARIO)</h2>", unsafe_allow_html=True)

        if not st.session_state.get("status_preprocess_ok", False):
            st.warning("⚠️ Jalankan menu *Preprocessing* dulu sampai normalisasi selesai.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        st.write("""Skenario Uji:
1) AHC Tanpa Seleksi Fitur
2) FCM Tanpa Seleksi Fitur
3) AHC Dengan Seleksi Fitur (Entropy Weighting)
4) FCM Dengan Seleksi Fitur (Entropy Weighting)
""")

        df_pre = st.session_state["df_preprocessed"].copy()
        df_scaled = st.session_state["df_scaled"].copy()
        X_norm = df_scaled.values

        colA, colB, colC = st.columns(3)
        with colA:
            k_min = st.number_input("k minimum", min_value=2, max_value=50, value=2, step=1)
        with colB:
            k_max = st.number_input("k maksimum", min_value=int(k_min), max_value=50, value=10, step=1)
        with colC:
            top_n_default = 5 if df_scaled.shape[1] >= 5 else df_scaled.shape[1]
            top_n = st.number_input("Top-N fitur Entropy", min_value=1, max_value=int(df_scaled.shape[1]), value=int(top_n_default), step=1)

        range_n_clusters = range(int(k_min), int(k_max) + 1)

        # pastikan entropy tersedia
        if st.session_state.get("df_entropy_weighted") is None or st.session_state.get("feature_ranking") is None:
            weights_entropy = entropy_weighting(df_scaled.values)
            feature_ranking = pd.DataFrame({
                "Feature": df_scaled.columns,
                "Weight": weights_entropy
            }).sort_values(by="Weight", ascending=False).reset_index(drop=True)
            df_entropy_weighted = df_scaled * weights_entropy

            st.session_state["weights_entropy"] = weights_entropy
            st.session_state["df_entropy_weighted"] = df_entropy_weighted
            st.session_state["feature_ranking"] = feature_ranking
        else:
            df_entropy_weighted = st.session_state["df_entropy_weighted"]
            feature_ranking = st.session_state["feature_ranking"].copy()

        selected_features = feature_ranking["Feature"].iloc[:int(top_n)].tolist()
        X_sub = df_entropy_weighted[selected_features].values
        st.caption(f"Fitur terpilih (Top-{int(top_n)} Entropy): {', '.join(selected_features)}")

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

                if len(np.unique(labels)) < 2:
                    sil = -1.0
                    dbi = np.inf
                    ch = -1.0
                else:
                    sil = float(silhouette_score(X, labels))
                    dbi = float(davies_bouldin_score(X, labels))
                    ch = float(calinski_harabasz_score(X, labels))

                sil_list.append(sil)
                dbi_list.append(dbi)
                ch_list.append(ch)
                labels_per_k[k] = labels

            return {"Skenario": name, "X": X, "is_fcm": is_fcm, "sil": sil_list, "dbi": dbi_list, "ch": ch_list, "labels_per_k": labels_per_k}

        if st.button("▶️ Jalankan Clustering (4 Skenario)"):
            scenario_results.append(eval_scenario("AHC Tanpa Seleksi Fitur", X_norm, is_fcm=False))
            scenario_results.append(eval_scenario("FCM Tanpa Seleksi Fitur", X_norm, is_fcm=True))
            scenario_results.append(eval_scenario("AHC Dengan Seleksi Fitur", X_sub, is_fcm=False))
            scenario_results.append(eval_scenario("FCM Dengan Seleksi Fitur", X_sub, is_fcm=True))

            all_ch = np.concatenate([np.array(s["ch"], dtype=float) for s in scenario_results])
            all_sil = np.concatenate([np.array(s["sil"], dtype=float) for s in scenario_results])
            all_dbi = np.concatenate([np.array(s["dbi"], dtype=float) for s in scenario_results])

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

            st.subheader("📈 Kurva Evaluasi per Skenario (Silhouette & DBI)")
            recap_rows = []
            ks = list(range_n_clusters)

            for s in scenario_results:
                name = s["Skenario"]
                sil = s["sil"]
                dbi = s["dbi"]

                best_k_sil = ks[int(np.argmax(sil))]
                best_sil = float(np.max(sil))

                best_k_dbi = ks[int(np.argmin(dbi))]
                best_dbi = float(np.min(dbi))

                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(plot_metric_curve(range_n_clusters, sil, best_k_sil, f"{name} - Silhouette", "Silhouette", color="blue"))
                with c2:
                    st.pyplot(plot_metric_curve(range_n_clusters, dbi, best_k_dbi, f"{name} - DBI", "DBI", color="black"))

                recap_rows.append({
                    "Skenario": name,
                    "BestK_Silhouette": best_k_sil,
                    "Silhouette_Max": best_sil,
                    "BestK_DBI": best_k_dbi,
                    "DBI_Min": best_dbi
                })

            df_recap = pd.DataFrame(recap_rows)
            st.dataframe(df_recap, use_container_width=True)

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

            st.subheader("📌 Rekapitulasi Hasil Clustering (Metode Terpilih)")

            chosen_scenario = next((s for s in scenario_results if s["Skenario"] == best_method), None)
            X_use = chosen_scenario["X"]
            is_fcm = chosen_scenario["is_fcm"]

            if not is_fcm:
                model_final = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
                labels_best = model_final.fit_predict(X_use)
                cntr_best = None
            else:
                cntr_best, u_best, *_ = fuzz.cluster.cmeans(
                    X_use.T, c=best_k, m=2, error=0.005, maxiter=1000, init=None, seed=42
                )
                labels_best = np.argmax(u_best, axis=0)

            df_hasil = df_pre.copy()
            df_hasil["Cluster"] = labels_best + 1

            izin_col = "surat Izin" if "surat Izin" in df_hasil.columns else None
            num_cols = [c for c in df_hasil.columns if c != "Cluster" and pd.api.types.is_numeric_dtype(df_hasil[c])]

            if num_cols:
                cluster_min = df_hasil.groupby("Cluster")[num_cols].min()
                cluster_max = df_hasil.groupby("Cluster")[num_cols].max()
                cluster_ranges = (cluster_min.astype(str) + " – " + cluster_max.astype(str)).reset_index()
            else:
                cluster_ranges = pd.DataFrame({"Cluster": sorted(df_hasil["Cluster"].unique())})

            if izin_col:
                izin_dist = df_hasil.groupby(["Cluster", izin_col]).size().unstack(fill_value=0).reset_index()
                cluster_summary = pd.merge(cluster_ranges, izin_dist, on="Cluster", how="left")
            else:
                cluster_summary = cluster_ranges

            st.write("✅ **Ringkasan tiap cluster (rentang min–max + distribusi kolom kategorikal jika ada)**")
            st.dataframe(cluster_summary, use_container_width=True)

            st.write("📌 **Jumlah anggota tiap cluster**")
            st.dataframe(
                df_hasil["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Jumlah Data"),
                use_container_width=True
            )

            st.subheader("📥 Download Hasil Clustering")
            st.session_state["df_clustered"] = df_hasil
            st.dataframe(df_hasil, use_container_width=True)

            csv_hasil = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Hasil Clustering (CSV)",
                data=csv_hasil,
                file_name="hasil_clustering.csv",
                mime="text/csv"
            )

            if "AHC" in best_method:
                centroids = np.array([X_use[labels_best == i].mean(axis=0) for i in range(best_k)])
                st.session_state["ahc_centroids"] = centroids
                st.session_state["fcm_cntr"] = None
            else:
                st.session_state["fcm_cntr"] = cntr_best
                st.session_state["ahc_centroids"] = None

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
