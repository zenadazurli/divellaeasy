#!/usr/bin/env python3
# divellaeasy_minimal.py - Versione FINALE con download funzionante

import os
import time
import requests
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

# ================ CONFIG =====================
DATASET_PATH = "dataset/dataset_speed.npz"
# Link diretto per download (NON è il link di visualizzazione)
DATASET_URL = "https://drive.usercontent.google.com/download?id=1fKdNNNN0tEh298RpNsubH9ajIiIzcQm1&export=download&confirm=t"
DIM = 64
REQUEST_TIMEOUT = 15

# ================ DATI ACCOUNT =====================
UID = "2288011"
COOKIE_SESIDS = "xa6LcV5CNv"
COOKIE_STRING = f"sesids={COOKIE_SESIDS}; user_id={UID}"

# ================ GLOBALS =====================
X_fast = None
y_fast = None
classes_fast = None

# ================ LOG =====================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ================ DOWNLOAD =====================
def download_dataset_if_missing():
    """Scarica il dataset SOLO se non esiste già"""
    dataset_path = Path(DATASET_PATH)
    
    # Se esiste già, non fare nulla
    if dataset_path.exists() and dataset_path.is_file():
        log(f"✅ Dataset già presente: {DATASET_PATH}")
        return True
    
    # Crea la cartella dataset se non esiste
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    log("📥 Download dataset da Google Drive...")
    log("⏳ Questa operazione può richiedere qualche minuto (file ~1GB)")
    
    try:
        # Download con requests (più robusto)
        response = requests.get(DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        # Verifica che non sia HTML
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            log("❌ Il link restituisce una pagina HTML, non il file")
            return False
        
        # Salva il file
        with open(dataset_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        log(f"✅ Download completato! File salvato in: {DATASET_PATH}")
        return True
        
    except Exception as e:
        log(f"❌ Errore download: {e}")
        return False

# ================ DATASET =====================
def load_dataset():
    global X_fast, y_fast, classes_fast
    
    if not download_dataset_if_missing():
        log("❌ Impossibile ottenere il dataset")
        return False
    
    try:
        data = np.load(DATASET_PATH, allow_pickle=True)
        X_fast = data["X"].astype(np.float32)
        y_fast = data["y"].astype(np.int32)
        
        if "classes" in data.files:
            classes = list(np.array(data["classes"], dtype=object).tolist())
        else:
            unique = sorted(list(set(int(x) for x in y_fast.tolist())))
            classes = [str(c) for c in unique]
        
        classes_fast = {i: classes[i] for i in range(len(classes))}
        log(f"✅ Dataset caricato: {X_fast.shape[0]} vettori")
        return True
    except Exception as e:
        log(f"❌ Errore caricamento dataset: {e}")
        return False

# ================ FEATURE EXTRACTION =====================
def get_features(img):
    img = cv2.resize(img, (DIM, DIM))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.flatten() / 255.0

def predict(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None
    features = get_features(img_crop)
    distances = np.linalg.norm(X_fast - features, axis=1)
    best_idx = np.argmin(distances)
    return classes_fast.get(int(y_fast[best_idx]), "errore")

# ================ CROP =====================
def crop_safe(img, coords):
    try:
        x1, y1, x2, y2 = map(int, coords.split(","))
    except:
        return None
    
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return img[y1:y2, x1:x2]

# ================ MAIN LOOP =====================
def main():
    log("=" * 50)
    log("🚀 Avvio DivellaEasy")
    
    if not load_dataset():
        return
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Cookie": COOKIE_STRING
    }
    session = requests.Session()
    
    while True:
        try:
            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                headers=headers, verify=False, timeout=REQUEST_TIMEOUT
            )
            if r.status_code != 200:
                log(f"❌ Status {r.status_code}")
                break
            
            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])
            
            if not urlid or not qpic or len(picmap) < 5:
                log("❌ Dati incompleti")
                break
            
            img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            crops = []
            for p in picmap:
                crops.append(crop_safe(img, p.get("coords", "")))
            
            labels = [predict(c) for c in crops]
            log(f"Labels: {labels}")
            
            seen = {}
            chosen_idx = None
            for i, label in enumerate(labels):
                if label and label != "errore":
                    if label in seen:
                        chosen_idx = seen[label]
                        break
                    seen[label] = i
            
            if chosen_idx is None:
                log("❌ Nessun duplicato")
                break
            
            time.sleep(seconds)
            word = picmap[chosen_idx]["value"]
            resp = session.get(
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                headers=headers, verify=False
            )
            
            if resp.json().get("warning") == "wrong_choice":
                log("❌ Wrong choice")
                break
            
            log(f"✅ OK")
            time.sleep(2)
            
        except Exception as e:
            log(f"❌ Errore: {e}")
            break

if __name__ == "__main__":
    main()
    log("🏁 Script terminato")


