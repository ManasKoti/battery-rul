"""Project-wide constants and paths."""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "MATR"            
PROCESSED_DIR = ROOT / "data" / "processed" / "MATR"  
CACHE_PATH = ROOT / "data" / "cache" / "matr_compact.pkl" 
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

NOMINAL_AH = 1.1         
EOL_FRACTION = 0.8       
EOL_AH = NOMINAL_AH * EOL_FRACTION  
 
V_GRID = np.linspace(3.5, 2.0, 1000) # the dataset's precomputed "Qdlin" curves are discharge capacity sampled at 1000 evenly spaced voltages from 3.5 V down to 2.0 V.

N_EARLY_CYCLES = 100      # let the model see cycles 1..100
SEED = 42