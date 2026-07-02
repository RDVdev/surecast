import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_phase(script_name):
    logging.info(f"========== Starting {script_name} ==========")
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        logging.info(f"========== Finished {script_name} ==========\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"FAILED: {script_name} exited with code {e.returncode}")
        sys.exit(1)

def main():
    phases = [
        "phase1_data_profiling.py",
        "phase2_sequence_construction.py",
        "phase3_feature_engineering.py",
        ["phase4_model_architecture.py", "--run_ablation"],
        "phase5_evaluation.py",
        "phase6_preference_alignment.py",
        "phase7_reporting_layer.py",
        "robustness_audit.py"
    ]
    
    for phase in phases:
        if isinstance(phase, list):
            logging.info(f"========== Starting {' '.join(phase)} ==========")
            subprocess.run([sys.executable] + phase, check=True)
            logging.info(f"========== Finished {' '.join(phase)} ==========\n")
        else:
            run_phase(phase)
        
    logging.info("ALL PHASES COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
