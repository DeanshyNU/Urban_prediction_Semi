"""
Wrapper to run original_code/run.py without modifying it.

What it does (no changes to original_code):
  1. Adds original_code/ to import path
  2. Imports original_code/run.py
  3. Overrides `run.path` to point at the V1 .mat files in code/data/
  4. Calls run.main()

This way the V1 baseline (V1 data + random temporal split) stays exactly as
the historical reference and can be re-run any time.
"""
import os
import sys

ORIGINAL_DIR = '/home/hhz6461/Urban_prediction_Semi/original_code'
DATA_DIR     = '/home/hhz6461/Urban_prediction_Semi/code/data'

# Prepend so original_code/* is what 'import data', 'import network', 'import utils' resolve to
sys.path.insert(0, ORIGINAL_DIR)

# Make sure the working directory has a place to write logs / checkpoints / param.pkl
# (the original run.py writes to cwd via './<modelName>_log' etc.)
output_dir = os.environ.get('V1_OUTPUT_DIR',
    f'/home/hhz6461/Urban_prediction_Semi/log/v1_baseline_'
    + os.environ.get('SLURM_JOB_ID', 'local'))
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

# Now import the original run.py
import run                                # original_code/run.py
import data                               # original_code/data.py — for sanity print

# Override the path inside run.py to point at where the V1 .mat files actually are
run.path = DATA_DIR
print(f"[wrapper] cwd            = {os.getcwd()}")
print(f"[wrapper] V1 data path    = {run.path}")
print(f"[wrapper] V1 .mat files  = {[f for f in os.listdir(DATA_DIR) if f.endswith('.mat') and ('StationMat' in f or 'AJM' in f or 'FeaturePatch_401' in f)]}")

# Run the original main() — completely unchanged
run.main()
