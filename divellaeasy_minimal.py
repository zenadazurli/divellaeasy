#!/usr/bin/env python3
# test_keepalive.py - Script di test per tenere attivo Render

import time
import requests
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def main():
    log("🚀 Script di test avviato - Versione SEMPLICE")
    log("Questo script non richiede dataset e resterà in esecuzione")
    
    counter = 0
    while True:
        counter += 1
        log(f"🔄 Ciclo #{counter} - Eseguo ping a Google...")
        
        try:
            # Ping semplice a Google per mostrare che funziona
            r = requests.get("https://www.google.com", timeout=10)
            log(f"✅ Google risponde con status {r.status_code}")
        except Exception as e:
            log(f"❌ Errore nel ping: {e}")
        
        log("😴 Attendo 60 secondi prima del prossimo ciclo...")
        time.sleep(60)

if __name__ == "__main__":
    main()







