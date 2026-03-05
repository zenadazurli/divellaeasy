#!/usr/bin/env python3
# solver.py - Versione con output visibile

import time
from datetime import datetime
import os

print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 SOLVER AVVIATO CON SUCCESSO!")
print(f"Directory: {os.getcwd()}")
print(f"File presenti: {os.listdir('.')}")

# Verifica dataset
dataset_path = "dataset/dataset_speed.npz"
if os.path.exists(dataset_path):
    size = os.path.getsize(dataset_path)
    print(f"✅ Dataset trovato: {size} bytes")
else:
    print(f"❌ Dataset non trovato")

counter = 0
while True:
    counter += 1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔁 Ciclo #{counter} - FUNZIONA!")
    time.sleep(30)
