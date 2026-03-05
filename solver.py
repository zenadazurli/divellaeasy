#!/usr/bin/env python3
# solver.py - Versione pulita per test

import time
from datetime import datetime
import os
import sys

print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 NUOVO SCRIPT AVVIATO")
print(f"Python: {sys.version}")
print(f"Directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# Verifica dataset
dataset_path = "dataset/dataset_speed.npz"
if os.path.exists(dataset_path):
    size = os.path.getsize(dataset_path)
    print(f"✅ Dataset trovato: {size} bytes")
else:
    print(f"❌ Dataset non trovato in {dataset_path}")

# Loop di test
counter = 0
while True:
    counter += 1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔁 Ciclo #{counter} - OK")
    time.sleep(30)
