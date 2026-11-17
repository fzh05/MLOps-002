import numpy as np
import pandas as pd
from src.fitur import pelajari_artefak, terapkan_fitur


def test_paritas_pelatihan_dan_produksi():
    # Ambil sampel data nyata (termasuk baris yang memiliki nilai kosong/NaN pada rata_saldo_90h)
    df = pd.read_csv("data/raw/kredit_sumatera.csv")
    df_latih = df[df["periode"] <= "2023-12"].head(200)
    df_uji = df[df["periode"] > "2024-06"].head(20)

    # 1. Pelajari parameter artefak dari data latih
    artefak = pelajari_artefak(df_latih)

    # 2. Simulasi pemrosesan batch (jalur pelatihan/evaluasi)
    X_batch = terapkan_fitur(df_uji, artefak)

    # 3. Simulasi pemrosesan satu per satu (jalur layanan produksi / API)
    hasil_per_baris = [
        terapkan_fitur(df_uji.iloc[[i]], artefak) for i in range(len(df_uji))
    ]
    X_produksi = pd.concat(hasil_per_baris, ignore_index=True)

    # Verifikasi arsitektural: urutan kolom dan nilai numerik wajib identik
    assert list(X_batch.columns) == list(X_produksi.columns), "Urutan kolom menyimpang!"
    assert not X_produksi.isna().any().any(), "Masih ada nilai NaN yang lolos di produksi!"
    assert np.allclose(X_batch.values, X_produksi.values), "Terdeteksi training-serving skew!"