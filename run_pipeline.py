"""
Pipeline orchestrator — runs all 7 phases sequentially.
BUG-4 fix: Unified error handling for both simple and list-type phases.
"""
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

PHASES = [
    "phase1_data_profiling.py",
    "phase2_sequence_construction.py",
    "phase3_feature_engineering.py",
    ["phase4_model_architecture.py", "--run_ablation"],
    "phase5_evaluation.py",
    "phase6_preference_alignment.py",
    "phase7_reporting_layer.py",
]

def run_phase(phase):
    """Run a single phase with unified error handling (BUG-4 fix)."""
    if isinstance(phase, list):
        cmd = [sys.executable] + phase
        phase_name = phase[0]
    else:
        cmd = [sys.executable, phase]
        phase_name = phase
    
    logging.info(f"\n{'='*60}")
    logging.info(f"RUNNING: {phase_name}")
    logging.info(f"{'='*60}\n")
    
    start = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        logging.info(f"\n✓ {phase_name} completed in {elapsed:.1f}s\n")
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        logging.error(f"\n✗ {phase_name} FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        sys.exit(e.returncode)
    except Exception as e:
        logging.error(f"\n✗ {phase_name} ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.info("═══════════════════════════════════════")
    logging.info("      Pipeline Orchestrator")
    logging.info("═══════════════════════════════════════\n")
    
    total_start = time.time()
    
    for phase in PHASES:
        run_phase(phase)
    
    total_elapsed = time.time() - total_start
    logging.info(f"\n{'='*60}")
    logging.info(f"ALL PHASES COMPLETE — Total time: {total_elapsed:.1f}s")
    logging.info(f"{'='*60}")
