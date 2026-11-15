"""Seleksi data ODIR-5K level mata dan pembagian train/val/test per pasien.

Modul ini tidak membutuhkan TensorFlow.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

LABEL_MAP = {
    "N": "Normal", "D": "Retinopati Diabetik",
    "G": "Glaukoma", "C": "Katarak",
    "A": "AMD", "H": "Hipertensi Retina",
    "M": "Miopia Patologis", "O": "Penyakit Lain",
}


def load_annotations(anno_file=config.ANNO_FILE) -> pd.DataFrame:
    """Baca file anotasi pasien ODIR-5K (satu baris per pasien)."""
    return pd.read_excel(anno_file)


def label_distribution(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Jumlah pasien untuk tiap kolom label (N, D, G, C, A, H, M, O)."""
    rows = []
    for col, name in LABEL_MAP.items():
        count = int(df_raw[col].sum()) if col in df_raw.columns else 0
        rows.append({"kode": col, "kelas": name, "jumlah_pasien": count})
    return pd.DataFrame(rows)


def parse_eye_label(keyword_str: str) -> int:
    """Kembalikan label untuk satu mata berdasarkan diagnostic keyword.

    Return:
      1  katarak  (keyword mengandung 'cataract')
      0  normal   (keyword HANYA 'normal fundus')
     -1  eksklusi (penyakit lain, lens dust tanpa cataract, keyword campuran, kosong)

    Catatan:
      'lens dust' tanpa cataract diberi -1, bukan 0. Lens dust menyebabkan
      kekeruhan lensa yang mirip katarak. Memasukkannya ke kelas Normal akan
      menambah noise pada batas keputusan model.
      'lens dust' bersama 'cataract' tetap berlabel 1.
    """
    kw = str(keyword_str).lower().strip()

    if "cataract" in kw:
        return 1
    if kw == "normal fundus":
        return 0
    return -1


def extract_records_eye_level(
    df_subset: pd.DataFrame,
    img_dir=config.IMG_DIR,
    exclude_normal_of_cataract_patient: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Ubah 1 baris pasien menjadi 2 baris mata (kiri dan kanan) dengan seleksi ketat.

    Parameter
    ---------
    exclude_normal_of_cataract_patient : bool
        Jika True, mata berlabel Normal milik pasien katarak (C=1) dikeluarkan.
        Mata yang tampak normal pada pasien katarak unilateral dapat memiliki
        perubahan subklinis. Memasukkannya ke kelas Normal berisiko mencemari
        kelas 0.
    """
    img_dir = Path(img_dir)
    records = []
    skip_other = 0    # penyakit lain, lens dust, keyword campuran
    skip_bocor = 0    # mata normal milik pasien C=1
    skip_nofile = 0   # file gambar tidak ditemukan

    for _, row in df_subset.iterrows():
        patient_is_cataract = int(row["C"]) == 1

        eyes = [
            (row["Left-Fundus"], row["Left-Diagnostic Keywords"], "L"),
            (row["Right-Fundus"], row["Right-Diagnostic Keywords"], "R"),
        ]
        for filename, keyword, side in eyes:
            if not (img_dir / str(filename)).exists():
                skip_nofile += 1
                continue

            label = parse_eye_label(keyword)

            if label == -1:
                skip_other += 1
                continue

            if exclude_normal_of_cataract_patient and label == 0 and patient_is_cataract:
                skip_bocor += 1
                continue

            records.append({
                "patient_id": int(row["ID"]),
                "filename": str(filename),
                "side": side,
                "label": label,
                "age": int(row.get("Patient Age", 0)),
                "sex": str(row.get("Patient Sex", "Unknown")).strip(),
                "keyword": str(keyword).lower().strip(),
            })

    df_out = pd.DataFrame(records).reset_index(drop=True)

    if verbose:
        n_cat = int((df_out["label"] == 1).sum())
        n_nor = int((df_out["label"] == 0).sum())
        print("=" * 52)
        print("  HASIL SELEKSI DATA LEVEL MATA")
        print("=" * 52)
        print(f"  Total mata lolos          : {len(df_out):>6}")
        print(f"  |- Katarak (label=1)      : {n_cat:>6}")
        print(f"  |- Normal  (label=0)      : {n_nor:>6}")
        if n_cat:
            print(f"  Rasio Normal:Katarak      :   1:{n_nor / n_cat:.2f}")
        print("-" * 52)
        print(f"  Dieksklusi penyakit lain  : {skip_other:>6}")
        print(f"  Dieksklusi label bocor    : {skip_bocor:>6}",
              "(nonaktif)" if not exclude_normal_of_cataract_patient else "")
        print(f"  Dieksklusi file hilang    : {skip_nofile:>6}")
        print("=" * 52)

        # Pasien katarak tidak boleh muncul di kelas Normal.
        cat_patients = set(df_out.loc[df_out["label"] == 1, "patient_id"])
        nor_patients = set(df_out.loc[df_out["label"] == 0, "patient_id"])
        overlap = cat_patients & nor_patients
        if overlap:
            print(f"\n  PERINGATAN: {len(overlap)} pasien muncul di KEDUA kelas.")
            print(f"  Contoh patient_id: {list(overlap)[:5]}")
        else:
            print("\n  OK: tidak ada tumpang-tindih pasien antar kelas.")

    return df_out


def add_image_path(df: pd.DataFrame, img_dir=config.IMG_DIR) -> pd.DataFrame:
    """Tambahkan kolom img_path (path lengkap) dari kolom filename."""
    df = df.copy()
    df["img_path"] = [str(Path(img_dir) / f) for f in df["filename"]]
    return df


def split_by_patient(df: pd.DataFrame, seed: int = config.SEED):
    """Bagi data per patient_id (bukan per baris).

    Setiap pasien punya dua mata. Pembagian per pasien mencegah mata kiri
    dan kanan pasien yang sama terpisah ke train dan test.
    """
    patient_ids = df["patient_id"].unique()
    # round() menghindari galat floating point: 1 - 0.70 = 0.30000000000000004,
    # yang dapat menggeser ukuran split karena train_test_split memakai ceil().
    test_tmp = round(1 - config.TRAIN_RATIO, 6)
    test_share = round(config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO), 6)
    ids_tr, ids_tmp = train_test_split(patient_ids, test_size=test_tmp, random_state=seed)
    ids_va, ids_te = train_test_split(ids_tmp, test_size=test_share, random_state=seed)

    return (df[df["patient_id"].isin(ids_tr)].reset_index(drop=True),
            df[df["patient_id"].isin(ids_va)].reset_index(drop=True),
            df[df["patient_id"].isin(ids_te)].reset_index(drop=True))


def make_splits(df_eye: pd.DataFrame, seed: int = config.SEED):
    """Split 70/15/15 per pasien, dilakukan terpisah untuk tiap kelas.

    Split dilakukan SEBELUM balancing. Val dan test tetap mempertahankan
    rasio kelas asli.
    """
    df_cat = df_eye[df_eye["label"] == 1].reset_index(drop=True)
    df_nor = df_eye[df_eye["label"] == 0].reset_index(drop=True)

    tr_cat, va_cat, te_cat = split_by_patient(df_cat, seed)
    tr_nor, va_nor, te_nor = split_by_patient(df_nor, seed)

    def _combine(a, b):
        return (pd.concat([a, b]).sample(frac=1, random_state=seed)
                .reset_index(drop=True))

    return _combine(tr_cat, tr_nor), _combine(va_cat, va_nor), _combine(te_cat, te_nor)


def split_summary(df_train, df_val, df_test) -> pd.DataFrame:
    """Ringkasan jumlah citra per split."""
    rows = []
    for name, df_ in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        nc = int((df_["label"] == 1).sum())
        nn = int((df_["label"] == 0).sum())
        rows.append({"split": name, "total": len(df_), "katarak": nc,
                     "normal": nn, "rasio_normal_katarak": round(nn / nc, 2)})
    return pd.DataFrame(rows)


def save_splits(df_train, df_val, df_test, out_dir=config.SPLITS_DIR) -> None:
    """Simpan daftar citra per split. Kolom img_path tidak ikut disimpan."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df_ in [("train", df_train), ("val", df_val), ("test", df_test)]:
        df_.drop(columns=["img_path"], errors="ignore").to_csv(
            out_dir / f"{name}.csv", index=False)


def load_splits(split_dir=config.SPLITS_DIR, img_dir=config.IMG_DIR):
    """Baca split yang disimpan notebook 01, lalu bentuk ulang kolom img_path."""
    split_dir = Path(split_dir)
    return tuple(add_image_path(pd.read_csv(split_dir / f"{n}.csv"), img_dir)
                 for n in ("train", "val", "test"))
