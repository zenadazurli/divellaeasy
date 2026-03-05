#!/usr/bin/env python3
# divellaeasy_minimal.py - Versione Hugging Face con token da ambiente

import os
import time
import requests
import numpy as np
import cv2
from datetime import datetime
from datasets import load_dataset

# ================ CONFIG =====================
DIM = 64
REQUEST_TIMEOUT = 15

# ================ DATI ACCOUNT =====================
UID = "2288011"
COOKIE_SESIDS = "KTeaAXDgWL"  # <-- SOSTITUISCI QUANDO SCADE
COOKIE_STRING = f"sesids={COOKIE_SESIDS}; user_id={UID}"

# ================ GLOBALS =====================
dataset = None
classes_fast = None

# ================ LOG =====================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ================ CARICAMENTO DATASET (HUGGING FACE) =====================
def load_dataset_hf():
    global dataset, classes_fast
    log("📥 Caricamento dataset da Hugging Face Hub...")
    
    # Legge il token dalla variabile d'ambiente (opzionale se dataset pubblico)
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token is None:
        log("⚠️ Token HF_TOKEN non trovato, tentativo senza autenticazione (dataset pubblico)")
        # Se il dataset è pubblico, non serve token
        token_param = None
    else:
        token_param = hf_token
    
    try:
        # Carica il dataset (con o senza token)
        dataset = load_dataset("zenadazurli/easyhits4u-dataset", split="train", token=token_param)
        log(f"✅ Dataset caricato: {len(dataset)} vettori")
        # Prepara le classi (per la conversione da indice a nome)
        class_names = dataset.features['y'].names
        classes_fast = {i: name for i, name in enumerate(class_names)}
        return True
    except Exception as e:
        log(f"❌ Errore caricamento dataset: {e}")
        return False

# ================ FUNZIONI DI FEATURE EXTRACTION =====================
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

    vettore = radiale + spaziale + [circularity, aspect_ratio] + hu
    return np.array(vettore, dtype=float)

def get_features(img):
    img_centrata = centra_figura(img)
    return estrai_descrittori(img_centrata)

# ================ PREDIZIONE =====================
def predict(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None
    features = get_features(img_crop)
    best_dist = float('inf')
    best_label_idx = None
    # Scorriamo il dataset in batch per non consumare troppa memoria
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        X_batch = np.array(batch['X'])  # (batch_size, 33)
        distances = np.linalg.norm(X_batch - features, axis=1)
        min_idx_batch = np.argmin(distances)
        if distances[min_idx_batch] < best_dist:
            best_dist = distances[min_idx_batch]
            best_label_idx = batch['y'][min_idx_batch]
    if best_label_idx is not None:
        return classes_fast.get(int(best_label_idx), "errore")
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
    log("🚀 Avvio DivellaEasy - Versione Hugging Face")
    if not load_dataset_hf():
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

















