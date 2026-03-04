#!/usr/bin/env python3
# divellaeasy_minimal.py - Versione con cookie da inserire

import os
import time
import requests
import numpy as np
import cv2
from datetime import datetime

# ================ CONFIG =====================
DATASET_PATH = "dataset/dataset_speed.npz"
DIM = 64
REQUEST_TIMEOUT = 15

# ================ DATI ACCOUNT =====================
UID = "2287667"  # <--- IL TUO USER ID (fisso, non cambia)

# 🔴 INSERISCI QUI IL COOKIE SESIDS (quando scade lo cambi)
COOKIE_SESIDS = "hvitsF06en"  # <-- Sostituisci con il nuovo valore quando scade

# Costruisce il cookie completo
COOKIE_STRING = f"sesids={COOKIE_SESIDS}; user_id={UID}"

# ================ GLOBALS =====================
X_fast = None
y_fast = None
classes_fast = None

# ================ LOG VELOCE =====================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ================ DATASET =====================
def load_dataset():
    global X_fast, y_fast, classes_fast
    data = np.load(DATASET_PATH, allow_pickle=True)
    X_fast = data["X"].astype(np.float32)
    y_fast = data["y"].astype(np.int32)
    classes = list(np.array(data.get("classes", range(len(np.unique(y_fast)))), dtype=object))
    classes_fast = {i: str(classes[i]) for i in range(len(classes))}
    log(f"✅ Dataset caricato: {X_fast.shape[0]} vettori")

# ================ FEATURE EXTRACTION MINIMAL =====================
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

# ================ MAIN LOOP =====================
def main():
    log("🚀 Avvio DivellaEasy Minimal")
    
    # Verifica che il cookie sia stato inserito
    if COOKIE_SESIDS == "WVwYztJ8Kk" and UID == "2287667":
        log("⚠️  Hai lasciato il cookie di esempio! Inserisci il cookie vero.")
        # Continuiamo lo stesso (tanto è quello giusto per ora)
    
    load_dataset()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Cookie": COOKIE_STRING
    }
    session = requests.Session()
    
    while True:
        try:
            # Richiedi surf
            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                headers=headers, verify=False, timeout=REQUEST_TIMEOUT
            )
            if r.status_code != 200:
                log(f"❌ Status {r.status_code} - Cookie forse scaduto?")
                log(f"💡 Aggiorna COOKIE_SESIDS nello script e riavvia")
                break
            
            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])
            
            if not urlid or not qpic or len(picmap) < 5:
                log("❌ Dati incompleti - Cookie forse scaduto?")
                break
            
            # Scarica immagine
            img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Riconosci le figure
            crops = []
            for p in picmap:
                x1, y1, x2, y2 = map(int, p["coords"].split(","))
                crops.append(img[y1:y2, x1:x2])
            
            labels = [predict(c) for c in crops]
            log(f"Labels: {labels}")
            
            # Trova il primo duplicato
            seen = {}
            chosen_idx = None
            for i, label in enumerate(labels):
                if label and label != "errore":
                    if label in seen:
                        chosen_idx = seen[label]
                        break
                    seen[label] = i
            
            if chosen_idx is None:
                log("❌ Nessun duplicato trovato")
                cv2.imwrite(f"errore_{qpic}.jpg", img)
                break
            
            # Aspetta e invia
            time.sleep(seconds)
            word = picmap[chosen_idx]["value"]
            resp = session.get(
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                headers=headers, verify=False
            )
            
            if resp.json().get("warning") == "wrong_choice":
                log("❌ Wrong choice")
                cv2.imwrite(f"errore_{qpic}.jpg", img)
                break
            
            log(f"✅ OK - {len(labels)} labels")
            time.sleep(2)
            
        except Exception as e:
            log(f"❌ Errore: {e}")
            break

if __name__ == "__main__":
    main()
    log("🏁 Script terminato - Cookie forse scaduto? Riavvia con cookie nuovo")