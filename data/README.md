# Folder data

Folder ini tidak berisi data. Dataset ODIR-5K dan cache citra tidak masuk Git.

## Dataset

Unduh dataset ODIR-5K (ODIR-2019, Peking University) dari sumber resminya. Atur strukturnya seperti ini:

```
data/
└── ODIR-5K/
    ├── Images/        citra fundus (*.jpg)
    └── data.xlsx      anotasi pasien (3.500 baris)
```

Kolom yang dipakai di `data.xlsx`: `ID`, `Patient Age`, `Patient Sex`, `Left-Fundus`, `Right-Fundus`, `Left-Diagnostic Keywords`, `Right-Diagnostic Keywords`, `N`, `D`, `G`, `C`, `A`, `H`, `M`, `O`.

Jika dataset berada di lokasi lain, atur variabel lingkungan sebelum membuka Jupyter:

```bash
# Linux / macOS
export ODIR_DATASET_PATH=/path/ke/ODIR-5K

# Windows (PowerShell)
$env:ODIR_DATASET_PATH = "D:\Dataset\ODIR-5K"
```

## Cache

Notebook 01 menulis citra hasil CLAHE ke `data/processed/` (`*_X.npy` dan `*_y.npy`). Notebook 02 dan 03 membaca cache ini.
