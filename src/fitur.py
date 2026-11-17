import pandas as pd

KOLOM_NUMERIK = [
    "usia", "tanggungan", "lama_usaha_tahun", "penghasilan_bulanan",
    "ragam_penghasilan", "plafon_diajukan", "tenor_bulan", "bunga_tahunan",
    "rasio_angsuran", "rata_saldo_90h", "jumlah_tunggakan_12b"
]
KOLOM_KATEGORIKAL = ["segmen", "wilayah", "kanal"]


def pelajari_artefak(df_latih: pd.DataFrame) -> dict:
    """Dijalankan HANYA pada data latih untuk mengunci median & urutan kolom."""
    df_tmp = df_latih.copy()
    df_tmp["rasio_angsuran"] = df_tmp["angsuran_bulanan"] / df_tmp["penghasilan_bulanan"]
    median_latih = df_tmp[KOLOM_NUMERIK].median().to_dict()
    X_encoded = pd.get_dummies(
        df_tmp[KOLOM_NUMERIK + KOLOM_KATEGORIKAL], columns=KOLOM_KATEGORIKAL
    )
    return {
        "median": median_latih,
        "urutan_kolom": list(X_encoded.columns),
    }


def terapkan_fitur(df: pd.DataFrame, artefak: dict) -> pd.DataFrame:
    """Satu definisi fitur bersama yang diimpor oleh pelatihan maupun produksi."""
    df_out = df.copy()
    df_out["rasio_angsuran"] = df_out["angsuran_bulanan"] / df_out["penghasilan_bulanan"]
    for col, val in artefak["median"].items():
        df_out[col] = df_out[col].fillna(val)
    X_encoded = pd.get_dummies(
        df_out[KOLOM_NUMERIK + KOLOM_KATEGORIKAL], columns=KOLOM_KATEGORIKAL
    )
    return X_encoded.reindex(columns=artefak["urutan_kolom"], fill_value=0)