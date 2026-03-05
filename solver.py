#!/usr/bin/env python3
# solver.py - Versione con output ogni 10 secondi

import time
from datetime import datetime
import os
import sys

print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 SOLVER PARTITO!", flush=True)
print(f"Directory: {os.getcwd()}", flush=True)
print(f"File: {os.listdir('.')}", flush=True)

counter = 0
while True:
    counter += 1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔁 Ciclo #{counter} - VIVO!", flush=True)
    time.sleep(10)
