#!/usr/bin/env python3
# divellaeasy_minimal.py - VERSIONE DI TEST

import time
from datetime import datetime

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Script di test avviato - Versione SEMPLICE")
    counter = 0
    while True:
        counter += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔁 Ciclo #{counter} - Script ancora in esecuzione")
        time.sleep(30)

if __name__ == "__main__":
    main()








