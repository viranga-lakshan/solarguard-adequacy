"""Shared path constants for the deprivation-aware monitoring pipeline.

Raw datasets are not duplicated into this repo -- they are read directly from
the existing downloads under RAW_ROOT. Only derived artefacts (parquet
frames, trained models, plots, result tables) are written under CODE_DIR.
"""

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
RAW_ROOT = Path(r"C:\Users\Viranga\Downloads\solar\data\raw")

MAFATE_RAW = RAW_ROOT / "mafate"
HEED_RAW = RAW_ROOT / "heed"
STAR_RAW = RAW_ROOT / "star"
JHARKHAND_RAW = RAW_ROOT / "jharkhand" / "vmehra813-solar_home_system_data_log-596c945" / "Script and Data"
SONG_KENYA_RAW = RAW_ROOT / "song_kenya"

DATA_DIR = CODE_DIR / "data"
MODELS_DIR = CODE_DIR / "models"
PLOTS_DIR = CODE_DIR / "plots"
RESULTS_DIR = CODE_DIR / "results"

for _d in (DATA_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
