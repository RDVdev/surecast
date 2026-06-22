from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app) # Enable CORS for React frontend

# Mapping phases to script files
PHASE_SCRIPTS = {
    "1": "phase1_data_profiling.py",
    "2": "phase2_sequence_construction.py",
    "3": "phase3_feature_engineering.py",
    "4": "phase4_model_architecture.py",
    "5": "phase5_evaluation.py",
    "6": "phase6_rlhf.py",
    "7": "phase7_reporting.py"
}

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check if the API is running and if the dataset is present."""
    has_dataset = os.path.exists("data/DataCoSupplyChainDataset.csv")
    return jsonify({
        "status": "online",
        "has_dataset": has_dataset
    })

@app.route('/api/run/<phase_id>', methods=['POST'])
def run_phase(phase_id):
    """Executes a specific pipeline phase and returns the logs."""
    if phase_id not in PHASE_SCRIPTS:
        return jsonify({"error": "Invalid phase ID"}), 400
        
    script_name = PHASE_SCRIPTS[phase_id]
    
    try:
        # Run the Python script using subprocess
        # Capture standard output and error
        result = subprocess.run(
            ["python3", script_name],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__name__))
        )
        
        # Combine stdout and stderr for the UI console
        logs = result.stdout
        if result.stderr:
            logs += "\n" + result.stderr
            
        success = result.returncode == 0
        
        return jsonify({
            "phase": phase_id,
            "success": success,
            "logs": logs
        })
        
    except Exception as e:
        return jsonify({
            "phase": phase_id,
            "success": False,
            "logs": f"Server execution error: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Run the Flask app on port 5005 to avoid macOS AirPlay Receiver port conflict on 5000
    app.run(port=5005, debug=True)
