#!/usr/bin/env python3
# divellaeasy_minimal.py - Versione FINALE con health check immediato e gestione picmap=None

import os
import time
import requests
import numpy as np
import cv2
import faiss
import json
import gc
import threading
from datetime import datetime
from datasets import load_dataset
from http.server import HTTPServer, BaseHTTPRequestHandler

# ================ CONFIG =====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "errori"
HEALTH_CHECK_PORT = int(os.environ.get('PORT', 10000))

# ================ DATI ACCOUNT =====================
UID = "2288934"
COOKIE_SESIDS = "Besr9LmUAc"  # <-- SOSTITUISCI QUANDO SCADE
COOKIE_STRING = f"sesids={COOKIE_SESIDS}; user_id={UID}"

# ================ GLOBALS =====================
dataset = None
classes_fast = None
faiss_index = None
vector_dim = 33
server_ready = False  # FLAG: indica quando il server è pronto

# ================ HEALTH CHECK SERVER (Risponde SUBITO) =====================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Silenzia i log del server

def run_health_server():
    """Avvia il server HTTP per gli health check in un thread separato"""
    global server_ready
    try:
        server = HTTPServer(('0.0.0.0', HEALTH_CHECK_PORT), HealthHandler)
        server_ready = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏥 Health check server avviato sulla porta {HEALTH_CHECK_PORT}")
        server.serve_forever()
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ ERRORE: impossibile avviare health check: {e}")
        server_ready = False

# Avvia il server in un thread separato (IMMEDIATAMENTE)
health_thread = threading.Thread(target=run_health_server, daemon=True)
health_thread.start()

# ================ ATTENDI CHE IL SERVER SIA PRONTO =====================
timeout = 10
while not server_ready and timeout > 0:
    time.sleep(0.5)
    timeout -= 0.5

# ================ LOG =====================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ================ CARICAMENTO DATASET (HUGGING FACE) =====================
def load_dataset_hf():
    global dataset, classes_fast, faiss_index
    log("📥 Caricamento dataset da Hugging Face Hub...")
    try:
        dataset = load_dataset("zenadazurli/easyhits4u-dataset", split="train", token=None)
        log(f"✅ Dataset caricato: {len(dataset)} vettori")
        class_names = dataset.features['y'].names
        classes_fast = {i: name for i, name in enumerate(class_names)}
        
        # COSTRUZIONE INDICE FAISS
        log("🔧 Costruzione indice FAISS...")
        X_list = []
        batch_size = 500
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            X_list.append(np.array(batch['X'], dtype=np.float32))
        
        X_all = np.vstack(X_list)
        log(f"📊 Vettori caricati: {X_all.shape}")
        
        # Indice FAISS con Product Quantization
        nlist = 100
        m = 3                # 33/3 = 11 dimensioni per sottovettore
        d = vector_dim
        
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        
        log("🏋️ Addestramento indice FAISS...")
        index.train(X_all)
        index.add(X_all)
        log(f"✅ Indice FAISS creato con {index.ntotal} vettori")
        faiss_index = index
        
        del X_list, X_all
        gc.collect()
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

# ================ PREDIZIONE CON FAISS =====================
def predict(img_crop):
    global faiss_index, classes_fast
    if img_crop is None or img_crop.size == 0:
        return None
    features = get_features(img_crop).astype(np.float32).reshape(1, -1)
    k = 1
    distances, indices = faiss_index.search(features, k)
    best_idx = indices[0][0]
    true_label_idx = dataset['y'][best_idx]
    return classes_fast.get(int(true_label_idx), "errore")

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
    log("=" * 50)
    log("🚀 Avvio DivellaEasy - Versione FAISS con Health Check Immediato")
    
    if not load_dataset_hf():
        log("❌ Impossibile proseguire senza dataset")
        return
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Cookie": COOKIE_STRING
    }
    session = requests.Session()
    captcha_counter = 0
    
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
            
            # GESTIONE MIGLIORATA: picmap potrebbe essere None
            if not urlid or not qpic or not picmap or len(picmap) < 5:
                log(f"❌ Dati incompleti - picmap={picmap} (tipo: {type(picmap).__name__})")
                log("   Cookie forse scaduto o risposta server vuota")
                # Non salviamo errore perché non abbiamo dati sufficienti
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
                salva_errore(qpic, img, picmap, labels, None, "nessun_duplicato", urlid)
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
                salva_errore(qpic, img, picmap, labels, chosen_idx, "wrong_choice", urlid)
                break
            
            captcha_counter += 1
            log(f"✅ OK - indice {chosen_idx} - Totale captcha: {captcha_counter}")
            
            if captcha_counter % 10 == 0:
                gc.collect()
                log(f"🧹 Garbage collection eseguita (captcha {captcha_counter})")
            
            time.sleep(2)
            
        except Exception as e:
            log(f"❌ Errore generico: {e}")
            import traceback
            log(traceback.format_exc())
            break

if __name__ == "__main__":
    main()
    log("🏁 Script terminato - In attesa di health check per riavvio")



























