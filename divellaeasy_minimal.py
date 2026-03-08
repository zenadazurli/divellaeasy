#!/usr/bin/env python3
# divellaeasy_minimal.py - Versione FINALE con refresh cookie via Bright Data

import os
import time
import requests
import numpy as np
import cv2
import json
import asyncio
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright

# ================ CONFIG =====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "errori"

# ================ DATI ACCOUNT EASYHITS =====================
EASYHITS_EMAIL = "clarabassoni2+borevunechi@gmail.com"
EASYHITS_PASSWORD = "DF45$!daza"

# ================ DATI BRIGHT DATA (da variabili d'ambiente) =====================
BRIGHTDATA_CUSTOMER_ID = os.environ.get('BRIGHTDATA_CUSTOMER_ID')
BRIGHTDATA_ZONE = "scraping_browser1"
BRIGHTDATA_ZONE_PASS = os.environ.get('BRIGHTDATA_ZONE_PASS')

if not BRIGHTDATA_CUSTOMER_ID or not BRIGHTDATA_ZONE_PASS:
    print("❌ ERRORE: Variabili d'ambiente BRIGHTDATA mancanti")
    print("Imposta BRIGHTDATA_CUSTOMER_ID e BRIGHTDATA_ZONE_PASS")
    exit(1)

# ================ GLOBALS =====================
X_fast = None
y_fast = None
classes_fast = None
session = None
current_user_id = None
current_sesids = None

# ================ LOG =====================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ================ REFRESH COOKIE VIA BRIGHT DATA =====================
async def refresh_cookies_via_brightdata():
    """
    Ottiene nuovi cookie da EasyHits4U via Bright Data
    Chiamato SOLO quando i cookie sono scaduti (dati incompleti)
    """
    log("🔄 Avvio refresh cookie via Bright Data...")
    
    ZONE_USER = f"brd-customer-{BRIGHTDATA_CUSTOMER_ID}-zone-{BRIGHTDATA_ZONE}"
    WS_ENDPOINT = f"wss://{ZONE_USER}:{BRIGHTDATA_ZONE_PASS}@brd.superproxy.io:9222"
    
    try:
        async with async_playwright() as p:
            log("🔌 Connessione a Bright Data...")
            browser = await p.chromium.connect_over_cdp(WS_ENDPOINT)
            context = browser.contexts[0]
            page = await context.new_page()
            
            log("🌐 Navigazione verso pagina di login...")
            await page.goto("https://www.easyhits4u.com/logon/", 
                          wait_until="domcontentloaded", 
                          timeout=60000)
            
            log("🛡️ Attesa risoluzione Turnstile...")
            await page.wait_for_timeout(10000)
            
            log("📝 Inserimento credenziali...")
            await page.evaluate(f'''
                document.querySelector('input[name="username"]').value = "{EASYHITS_EMAIL}";
                document.querySelector('input[name="password"]').value = "{EASYHITS_PASSWORD}";
                const event = new Event('input', {{ bubbles: true }});
                document.querySelector('input[name="password"]').dispatchEvent(event);
            ''')
            
            log("🖱️ Invio form...")
            await page.evaluate('''
                document.querySelector('form[action="/logon/"] input[type="submit"]').click();
            ''')
            
            log("⏳ Attesa risposta...")
            await page.wait_for_timeout(10000)
            
            cookies = await context.cookies()
            user_id = next((c['value'] for c in cookies if c['name'] == 'user_id'), None)
            sesids = next((c['value'] for c in cookies if c['name'] == 'sesids'), None)
            
            await browser.close()
            
            if user_id and sesids:
                log(f"✅ Nuovi cookie ottenuti: user_id={user_id}, sesids={sesids[:10]}...")
                return user_id, sesids
            else:
                log("❌ Cookie non trovati nella risposta")
                return None, None
                
    except Exception as e:
        log(f"❌ Errore refresh cookie: {e}")
        return None, None

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

# ================ CARICAMENTO DATASET =====================
def load_dataset():
    global X_fast, y_fast, classes_fast
    
    DATASET_PATH = "dataset/dataset_speed.npz"
    if not os.path.exists(DATASET_PATH):
        log("❌ Dataset non trovato, impossibile proseguire")
        return False
    
    data = np.load(DATASET_PATH, allow_pickle=True)
    X_fast = data["X"].astype(np.float32)
    y_fast = data["y"].astype(np.int32)
    
    if "classes" in data.files:
        classes = list(np.array(data["classes"], dtype=object).tolist())
    else:
        unique = sorted(list(set(int(x) for x in y_fast.tolist())))
        classes = [str(c) for c in unique]
    
    classes_fast = {i: classes[i] for i in range(len(classes))}
    log(f"✅ Dataset caricato: {X_fast.shape[0]} vettori, {len(classes)} classi")
    return True

# ================ PREDIZIONE =====================
def predict(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None
    features = get_features(img_crop)
    distances = np.linalg.norm(X_fast - features, axis=1)
    best_idx = np.argmin(distances)
    return classes_fast.get(int(y_fast[best_idx]), "errore")

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

# ================ SALVATAGGIO ERRORI =====================
def salva_errore(qpic, img, picmap, labels, chosen_idx, motivo, urlid=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(ERRORI_DIR, f"{timestamp}_{qpic}")
    os.makedirs(folder, exist_ok=True)
    
    full_path = os.path.join(folder, "full.jpg")
    cv2.imwrite(full_path, img)
    
    crops_saved = []
    for i, p in enumerate(picmap):
        crop = crop_safe(img, p.get("coords", ""))
        if crop is not None and crop.size > 0:
            crop_filename = f"crop_{i+1}.jpg"
            crop_path = os.path.join(folder, crop_filename)
            cv2.imwrite(crop_path, crop)
            crops_saved.append(crop_filename)
        else:
            crops_saved.append(None)
    
    metadata = {
        "timestamp": timestamp,
        "qpic": qpic,
        "urlid": urlid,
        "motivo": motivo,
        "labels_predette": labels,
        "chosen_idx": chosen_idx,
        "picmap_values": [p.get("value") for p in picmap] if picmap else [],
        "crops_salvati": crops_saved,
        "immagine_intera": "full.jpg"
    }
    
    metadata_path = os.path.join(folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    log(f"📁 Errore salvato in {folder}")

# ================ MAIN LOOP =====================
def main():
    global session, current_user_id, current_sesids
    
    log("=" * 50)
    log("🚀 Avvio DivellaEasy - Versione FINALE")
    
    # Carica dataset
    if not load_dataset():
        log("❌ Impossibile proseguire senza dataset")
        return
    
    # Ottieni cookie iniziali
    log("🍪 Richiedo cookie iniziali a Bright Data...")
    user_id, sesids = asyncio.run(refresh_cookies_via_brightdata())
    if not user_id or not sesids:
        log("❌ Impossibile ottenere cookie iniziali")
        return
    
    current_user_id = user_id
    current_sesids = sesids
    COOKIE_STRING = f"sesids={current_sesids}; user_id={current_user_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Cookie": COOKIE_STRING
    }
    session = requests.Session()
    session.headers.update(headers)
    
    captcha_counter = 0
    
    while True:
        try:
            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                verify=False, timeout=REQUEST_TIMEOUT
            )
            
            if r.status_code != 200:
                log(f"⚠️ Status {r.status_code}")
                time.sleep(5)
                continue
            
            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])
            
            # SCENARIO 2: DATI INCOMPLETI = COOKIE SCADUTI
            if not urlid or not qpic or not picmap or len(picmap) < 5:
                log("❌ DATI INCOMPLETI - Cookie scaduti")
                log("🔄 Richiedo nuovi cookie...")
                new_uid, new_ses = asyncio.run(refresh_cookies_via_brightdata())
                if new_uid and new_ses:
                    current_user_id = new_uid
                    current_sesids = new_ses
                    session.headers.update({"Cookie": f"sesids={current_sesids}; user_id={current_user_id}"})
                    log("✅ Cookie aggiornati, continuo...")
                    continue
                else:
                    log("❌ Impossibile aggiornare cookie, fermo script")
                    break
            
            # SCENARIO 1: RICONOSCIMENTO
            img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
            labels = [predict(c) for c in crops]
            log(f"Labels: {labels}")
            
            # Trova duplicati
            seen = {}
            chosen_idx = None
            for i, label in enumerate(labels):
                if label and label != "errore":
                    if label in seen:
                        chosen_idx = seen[label]
                        break
                    seen[label] = i
            
            if chosen_idx is None:
                log("❌ NESSUN DUPLICATO - Errore riconoscimento")
                salva_errore(qpic, img, picmap, labels, None, "nessun_duplicato", urlid)
                log("🛑 Script fermato per analisi errori")
                break
            
            time.sleep(seconds)
            word = picmap[chosen_idx]["value"]
            resp = session.get(
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                verify=False
            )
            
            if resp.json().get("warning") == "wrong_choice":
                log("❌ WRONG CHOICE - Errore riconoscimento")
                salva_errore(qpic, img, picmap, labels, chosen_idx, "wrong_choice", urlid)
                log("🛑 Script fermato per analisi errori")
                break
            
            captcha_counter += 1
            log(f"✅ OK - indice {chosen_idx} - Totale captcha: {captcha_counter}")
            
            time.sleep(2)
            
        except Exception as e:
            log(f"❌ Errore generico: {e}")
            time.sleep(5)
            continue

if __name__ == "__main__":
    main()




























