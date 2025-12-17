        # =====================================================
        # REKAPITULASI HASIL CLUSTERING (berdasarkan metode terpilih)
        # tampilkan setelah "Metode terbaik berdasarkan ..."
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

        # Tentukan fitur yang digunakan untuk rekap:
        # - jika tanpa seleksi -> pakai semua fitur df_new
        # - jika seleksi -> pakai selected_features (5 fitur teratas)
        if "Dengan Seleksi Fitur" in best_method:
            recap_features = selected_features[:]  # 5 fitur teratas (entropy)
            df_recap_source = df[recap_features].copy()
        else:
            recap_features = list(df_new.columns)  # semua fitur numeric + surat izin sudah encoded
            df_recap_source = df_new.copy()

        # Hitung label final sesuai metode terbaik dan best_k
        if not is_fcm:
            model_final = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
            labels_best = model_final.fit_predict(X_use)
        else:
            cntr_best, u_best, *_ = fuzz.cluster.cmeans(
                X_use.T, c=best_k, m=2, error=0.005, maxiter=1000, init=None, seed=42
            )
            labels_best = np.argmax(u_best, axis=0)

        # buat 1-based cluster
        df_rekap = df_recap_source.copy()
        df_rekap["Cluster"] = labels_best + 1

        # Pastikan "surat Izin" ikut direkap kalau ada
        izin_col = "surat Izin" if "surat Izin" in df_rekap.columns else None

        # Pisahkan fitur numerik (untuk min-max)
        num_cols = [c for c in df_rekap.columns if c not in ["Cluster", izin_col] and pd.api.types.is_numeric_dtype(df_rekap[c])]

        # 1) Rentang min-max untuk numerik
        if num_cols:
            cluster_min = df_rekap.groupby("Cluster")[num_cols].min()
            cluster_max = df_rekap.groupby("Cluster")[num_cols].max()
            cluster_ranges = (cluster_min.astype(str) + " – " + cluster_max.astype(str)).reset_index()
        else:
            cluster_ranges = pd.DataFrame({"Cluster": sorted(df_rekap["Cluster"].unique())})

        # 2) Distribusi surat izin (jika ada)
        if izin_col:
            izin_dist = (
                df_rekap.groupby(["Cluster", izin_col])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            cluster_summary = pd.merge(cluster_ranges, izin_dist, on="Cluster", how="left")
        else:
            cluster_summary = cluster_ranges

        # tampilkan tabel rekap
        st.write("✅ **Ringkasan tiap cluster (rentang min–max + distribusi surat izin)**")
        st.dataframe(cluster_summary, use_container_width=True)

        # optional: tampilkan jumlah anggota cluster
        st.write("📌 **Jumlah anggota tiap cluster**")
        st.dataframe(
            df_rekap["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Jumlah Data"),
            use_container_width=True
        )

        # simpan ke session_state agar bisa dipakai di halaman lain kalau mau
        st.session_state["best_labels_1based"] = (labels_best + 1)
        st.session_state["cluster_summary_best"] = cluster_summary
