import subprocess
import sys
import os

SEED = 42

os.makedirs('outputs/realnvp_outputs', exist_ok=True)
os.makedirs('outputs/wgan_outputs', exist_ok=True)

scripts = ['real_nvp.py', 'wgan_gp.py']

for script in scripts:
    subprocess.run([sys.executable, script, '--seed', str(SEED)])