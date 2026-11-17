import hashlib
import json
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd


def hitung_hash_file(path_file: str | Path) -> str:
    p = Path(path_file)
    if not p.exists():
        return "FILE_TIDAK_DITEMUKAN"
    sha256 = hashlib.sha256()
    with p.open("rb") as f:
        for blok in iter(lambda: f.read(65536), b""):
            sha256.update(blok)
    return sha256.hexdigest()[:16]


def simpan_checkpoint(
    nama_tahap: str,
    variabel_dict: dict,
    model=None,
    path_data: str = "data/raw/kredit_sumatera.csv",
    folder_simpan: str = "artifacts/checkpoints",
) -> Path:
    tujuan = Path(folder_simpan) / nama_tahap
    tujuan.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tahap": nama_tahap,
        "waktu_simpan": datetime.now().isoformat(timespec="seconds"),
        "hash_dataset": hitung_hash_file(path_data),
        "variabel_tercatat": {k: repr(v) for k, v in variabel_dict.items()},
    }
    (tujuan / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    payload = {"variabel": variabel_dict, "model": model}
    joblib.dump(payload, tujuan / "state_lengkap.joblib")

    print(f"[CHECKPOINT TERSIMPAN] '{nama_tahap}' -> {tujuan}")
    print(f"  - Hash Dataset : {metadata['hash_dataset']}")
    print(f"  - Variabel     : {list(variabel_dict.keys())}")
    return tujuan


def muat_checkpoint(
    nama_tahap: str,
    path_data: str = "data/raw/kredit_sumatera.csv",
    folder_simpan: str = "artifacts/checkpoints",
) -> tuple[dict, object]:
    tujuan = Path(folder_simpan) / nama_tahap
    metadata = json.loads((tujuan / "metadata.json").read_text(encoding="utf-8"))
    payload = joblib.load(tujuan / "state_lengkap.joblib")

    hash_sekarang = hitung_hash_file(path_data)
    if hash_sekarang != metadata["hash_dataset"]:
        print(f"[PERINGATAN] Hash dataset berubah! ({metadata['hash_dataset']} -> {hash_sekarang})")
    else:
        print(f"[OK] Dataset cocok dengan checkpoint '{nama_tahap}' ({hash_sekarang})")

    return payload["variabel"], payload["model"]


if __name__ == "__main__":
    csv_path = Path("data/raw/kredit_sumatera.csv")
    if not csv_path.exists():
        print("Dataset belum ada. Jalankan: python buat_dataset.py --keluaran data/raw --seed 42")
    else:
        df = pd.read_csv(csv_path)
        var_awal = {
            "seed": 42,
            "jumlah_baris": len(df),
            "daftar_kolom": list(df.columns),
            "batas_latih": "2023-12",
            "batas_uji": "2024-01",
            "tingkat_gagal_bayar_total": round(float(df["gagal_bayar"].mean()), 4),
        }
        simpan_checkpoint("prak01_selesai", variabel_dict=var_awal, model=None)