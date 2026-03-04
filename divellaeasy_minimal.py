#!/usr/bin/env python3
# divellaeasy_minimal.py - Versione FINALE con feature extraction 33 dimensioni

import os
import time
import requests
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

# ================ CONFIG =====================
DATASET_PATH = "dataset/dataset_speed.npz"
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

# ================ DOWNLOAD DATASET =====================
def download_dataset_if_missing():
    dataset_path = Path(DATASET_PATH)
    if dataset_path.exists() and dataset_path.is_file():
        log(f"✅ Dataset già presente: {DATASET_PATH}")
        return True
    
    log(f"📁 Creazione cartella: {dataset_path.parent}")
    os.makedirs(dataset_path.parent, exist_ok=True)
    
    if not os.path.exists(dataset_path.parent):
        log(f"❌ Impossibile creare la cartella {dataset_path.parent}")
        return False
    
    log("📥 Download dataset da Google Drive...")
    log("⏳ Questa operazione può richiedere qualche minuto (file ~1GB)")
    
    try:
        response = requests.get(DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            log("❌ Il link restituisce una pagina HTML, non il file")
            return False
        
        with open(dataset_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if dataset_path.exists():
            log(f"✅ Download completato! File salvato in: {DATASET_PATH}")
            return True
        else:
            log("❌ File non trovato dopo il download")
            return False
    except Exception as e:
        log(f"❌ Errore download: {e}")
        return False

# ================ FUNZIONI DI FEATURE EXTRACTION (33 DIMENSIONI) =====================
def centra_figura(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (DIM, DIM))
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 0.0
    aspect_ratio = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri != 0:
            circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0.0

    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()

    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    raggi = [int(min(h,w)*r) for r in (0.2, 0.4, 0.6, 0.8)]
    radiale = []
    for r in raggi:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (cx,cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m/255.0 for m in mean])

    spaziale = []
    quadranti = [(0,0,cx,cy), (cx,0,w,cy), (0,cy,cx,h), (cx,cy,w,h)]
    for (x1,y1,x2,y2) in quadranti:
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m/255.0 for m in mean])

    # 12 (radiale) + 12 (spaziale) + 1 + 1 + 7 = 33 feature
    vettore = radiale + spaziale + [circularity, aspect_ratio] + hu
    return np.array(vettore, dtype=float)

def get_features(img):
    img_centrata = centra_figura(img)
    return estrai_descrittori(img_centrata)

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
        log(f"✅ Dataset caricato: {X_fast.shape[0]} vettori, {X_fast.shape[1]} feature, {len(classes)} classi")
        return True
    except Exception as e:
        log(f"❌ Errore caricamento dataset: {e}")
        return False

# ================ PREDIZIONE =====================
def predict(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None
    try:
        features = get_features(img_crop)
        # Verifica dimensioni (debug)
        if len(features) != 33:
            log(f"⚠️ Attenzione: feature ha {len(features)} dimensioni, attese 33")
            if len(features) > 33:
                features = features[:33]
            else:
                features = np.pad(features, (0, 33 - len(features)), 'constant')
        distances = np.linalg.norm(X_fast - features, axis=1)
        best_idx = np.argmin(distances)
        return classes_fast.get(int(y_fast[best_idx]), "errore")
    except Exception as e:
        log(f"Errore in predict: {e}")
        return "errore"

# ================ CROP SICURO =====================
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
    log("🚀 Avvio DivellaEasy - Feature extraction 33 dimensioni")
    if not load_dataset():
        log("❌ Impossibile proseguire senza dataset")
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
                log(f"❌ Status {r.status_code} - Cookie forse scaduto?")
                break
            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])
            if not urlid or not qpic or len(picmap) < 5:
                log("❌ Dati incompleti - Cookie forse scaduto?")
                break
            img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
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
                log("❌ Nessun duplicato trovato")
                cv2.imwrite(f"errore_{qpic}.jpg", img)
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
                cv2.imwrite(f"errore_{qpic}.jpg", img)
                break
            log(f"✅ OK - indice {chosen_idx}")
            time.sleep(2)
        except Exception as e:
            log(f"❌ Errore: {e}")
            break

if __name__ == "__main__":
    main()
    log("🏁 Script terminato")





