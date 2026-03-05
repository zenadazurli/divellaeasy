#!/usr/bin/env python3
# divellaeasy_minimal.py - Versione con logging estremo su file

import os
import sys
import time
from datetime import datetime

# ================ CONFIG LOGGING =====================
LOG_DIR = "/opt/render/project/src/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log_to_file(msg):
    """Scrive messaggio su file di log con timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {msg}\n")
    # Stampa anche a video per i log di Render
    print(f"[{timestamp}] {msg}")

# Inizio logging
log_to_file("🚀 SCRIPT AVVIATO")
log_to_file(f"Python version: {sys.version}")
log_to_file(f"Current directory: {os.getcwd()}")
log_to_file(f"Files in current dir: {os.listdir('.')}")

# ================ TEST IMPORTAZIONI =====================
log_to_file("--- Test importazioni ---")

try:
    import cv2
    log_to_file("✅ OpenCV importato")
    log_to_file(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    log_to_file(f"❌ OpenCV ERRORE: {e}")

try:
    import numpy as np
    log_to_file("✅ NumPy importato")
    log_to_file(f"NumPy version: {np.__version__}")
except Exception as e:
    log_to_file(f"❌ NumPy ERRORE: {e}")

try:
    import requests
    log_to_file("✅ Requests importato")
    log_to_file(f"Requests version: {requests.__version__}")
except Exception as e:
    log_to_file(f"❌ Requests ERRORE: {e}")

try:
    from pathlib import Path
    log_to_file("✅ Pathlib importato")
except Exception as e:
    log_to_file(f"❌ Pathlib ERRORE: {e}")

# ================ VERIFICA DATASET =====================
log_to_file("--- Verifica dataset ---")

DATASET_PATH = "dataset/dataset_speed.npz"
dataset_full_path = os.path.join(os.getcwd(), DATASET_PATH)
log_to_file(f"Cerco dataset in: {dataset_full_path}")

if os.path.exists(DATASET_PATH):
    log_to_file(f"✅ File dataset trovato")
    file_size = os.path.getsize(DATASET_PATH)
    log_to_file(f"Dimensione: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
else:
    log_to_file(f"❌ File dataset NON trovato")

# ================ TENTATIVO CARICAMENTO DATASET =====================
log_to_file("--- Tentativo caricamento dataset ---")

try:
    data = np.load(DATASET_PATH, allow_pickle=True)
    log_to_file("✅ np.load riuscito")
    
    if "X" in data.files:
        X = data["X"]
        log_to_file(f"✅ X trovato con shape: {X.shape}")
    else:
        log_to_file("❌ X non trovato nel file")
        
    if "y" in data.files:
        y = data["y"]
        log_to_file(f"✅ y trovato con shape: {y.shape}")
    else:
        log_to_file("❌ y non trovato nel file")
        
    if "classes" in data.files:
        classes = data["classes"]
        log_to_file(f"✅ classes trovato con shape: {classes.shape if hasattr(classes, 'shape') else len(classes)}")
    else:
        log_to_file("ℹ️ classes non trovato (non grave)")
        
except Exception as e:
    log_to_file(f"❌ ERRORE caricamento dataset: {e}")
    import traceback
    log_to_file(traceback.format_exc())

log_to_file("=" * 50)
log_to_file(f"Log completato. File salvato in: {LOG_FILE}")
log_to_file("=" * 50)

# ================ SE TUTTO OK, PROCEDO CON LO SCRIPT =====================
try:
    # Qui puoi mettere il resto del tuo script originale
    # Per ora lasciamo un loop di test
    log_to_file("--- Avvio loop principale ---")
    counter = 0
    while True:
        counter += 1
        log_to_file(f"🔄 Ciclo #{counter} - Script in esecuzione")
        time.sleep(60)
except Exception as e:
    log_to_file(f"❌ ERRORE nel loop principale: {e}")











