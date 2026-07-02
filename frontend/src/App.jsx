import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5005/api';

const PHASES = [
  { id: '1', name: 'Phase 1: Profiling', desc: 'Clean and format raw dataset' },
  { id: '2', name: 'Phase 2: Sequences', desc: 'Construct multi-step tensors' },
  { id: '3', name: 'Phase 3: Features', desc: 'Generate domain interactions' },
  { id: '4', name: 'Phase 4: Architecture', desc: 'Train DL/ML branches' },
  { id: '5', name: 'Phase 5: Evaluation', desc: 'Rolling CV & Calibration' },
  { id: '6', name: 'Phase 6: Alignment', desc: 'RLHF Preference Tuning' },
  { id: '7', name: 'Phase 7: Reporting', desc: 'Generate Template Summaries' },
];

function App() {
  const [serverStatus, setServerStatus] = useState('checking');
  const [hasDataset, setHasDataset] = useState(false);
  const [activePhase, setActivePhase] = useState(null);
  const [logs, setLogs] = useState("System Ready. Connect to SUREcast backend...\n");
  const [isRunning, setIsRunning] = useState(false);
  const terminalRef = useRef(null);

  useEffect(() => {
    checkServerStatus();
    // Poll server status every 10s
    const interval = setInterval(checkServerStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const checkServerStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/status`);
      setServerStatus('online');
      setHasDataset(res.data.has_dataset);
      if (!res.data.has_dataset && !logs.includes('WARNING')) {
        appendLog('WARNING: DataCoSupplyChainDataset.csv not found in data/ folder.');
      }
    } catch (err) {
      setServerStatus('offline');
    }
  };

  const appendLog = (text) => {
    setLogs((prev) => prev + text + '\n');
  };

  const runPhase = async (phaseId) => {
    if (serverStatus !== 'online') {
      appendLog('[ERROR] Cannot run: Python backend server is offline.');
      return;
    }
    
    setIsRunning(true);
    setActivePhase(phaseId);
    appendLog(`\n> Executing Phase ${phaseId}...`);
    
    try {
      const res = await axios.post(`${API_URL}/run/${phaseId}`);
      appendLog(res.data.logs);
      if (res.data.success) {
        appendLog(`[SUCCESS] Phase ${phaseId} completed.`);
      } else {
        appendLog(`[FAILED] Phase ${phaseId} exited with errors.`);
      }
    } catch (err) {
      appendLog(`[CRITICAL] Network error communicating with backend: ${err.message}`);
    } finally {
      setIsRunning(false);
      setActivePhase(null);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>SURE<span className="title-gradient">cast</span> Dashboard</h1>
        <div className="server-status">
          <div className={`status-dot ${serverStatus}`}></div>
          Backend: {serverStatus.toUpperCase()} 
          {serverStatus === 'online' && ` | Data: ${hasDataset ? 'READY' : 'MISSING'}`}
        </div>
      </header>

      <div className="controls-sidebar">
        <div className="glass-panel">
          <h2 style={{ marginBottom: '16px', fontSize: '1.2rem', fontWeight: 600 }}>Pipeline Execution</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {PHASES.map((phase) => (
              <div 
                key={phase.id} 
                className={`glass-panel phase-card ${activePhase === phase.id ? 'active' : ''}`}
                style={{ padding: '12px 16px', background: 'rgba(15, 23, 42, 0.4)' }}
              >
                <div className="phase-info">
                  <h3>{phase.name}</h3>
                  <p>{phase.desc}</p>
                </div>
                <button 
                  className="play-btn"
                  onClick={() => runPhase(phase.id)}
                  disabled={isRunning}
                >
                  {activePhase === phase.id ? 'Running...' : 'Run'}
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="console-area">
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '24px' }}>
          <div className="console-header">
            <h2>System Terminal</h2>
            <button 
              className="play-btn" 
              onClick={() => setLogs("System Ready...\n")}
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
            >
              Clear
            </button>
          </div>
          <div className="terminal-window" ref={terminalRef}>
            {logs}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
