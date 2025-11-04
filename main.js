// main.js (root alongside /worklets, /workers, /libs, /onnx, /artifacts, /utils)

let audioCtx;
let workletNode;
let manager;

let managerReady = false;
let rnnReady = false;
let isPlaying = false;

// --- UI elements (from your index.html card) ---
const playBtn    = document.getElementById('playBtn');
const stopBtn    = document.getElementById('stopBtn');
const fillSlider = document.getElementById('fillSlider');
const fillValue  = document.getElementById('fillValue');

// Helper to read current FillLevel (as number 0..1)
function currentFill() {
  return Number(fillSlider?.value ?? 0.5);
}

// Reflect slider value text
function updateFillLabel() {
  if (fillValue) fillValue.textContent = currentFill().toFixed(2);
}
updateFillLabel();

// ------------------ Boot ------------------
async function boot() {
  // 1) AudioContext + worklet
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  // Load the worklet module
  await audioCtx.audioWorklet.addModule('/worklets/WaterFillRNNWorklet.js');

  // Create the worklet node; its processor name is 'water-fill-rnn'
  workletNode = new AudioWorkletNode(audioCtx, 'water-fill-rnn', {
    processorOptions: {
      sampleRate: audioCtx.sampleRate,
    },
  });

  // Directly to destination for this minimal test
  workletNode.connect(audioCtx.destination);

  // 2) Manager worker (spawns RNN worker internally)
  manager = new Worker('/workers/manager-worker.js?cb=' + Date.now(), { type: 'module' });

  // --- Manager → Main
  manager.onmessage = (ev) => {
    const msg = ev.data;
    if (!msg || !msg.type) return;

    if (msg.type === 'manager-ready') {
      managerReady = true;
      // If we’re already playing, prime immediately
      if (isPlaying && rnnReady) {
        manager.postMessage({ type: 'needHop', fillLevel: currentFill() });
      }
      return;
    }

    if (msg.type === 'ready') {
      // “RNN (via manager) ready”
      rnnReady = true;
      // Prime once both ends are ready
      if (isPlaying && managerReady) {
        manager.postMessage({ type: 'needHop', fillLevel: currentFill() });
      }
      return;
    }

    if (msg.type === 'audioHop') {
      // Send audio to the worklet ring buffer
      // (Transfer the buffer for zero-copy)
      workletNode.port.postMessage(
        { type: 'audioHop', samples: msg.samples, sr: msg.sr },
        [msg.samples.buffer]
      );
      return;
    }

    if (msg.type === 'error') {
      console.warn('[main] worker error:', msg.error);
      return;
    }
  };

  // --- Main → Manager (init)
  manager.postMessage({ type: 'init', targetSr: audioCtx.sampleRate });

  // --- Worklet → Main
  workletNode.port.onmessage = (ev) => {
    const msg = ev.data;
    if (!msg || !msg.type) return;

    if (msg.type === 'needHop') {
      // Worklet asks for the next hop when its circular buffer is low.
      // Always include the CURRENT FillLevel so conditioning is JIT.
      if (managerReady) {
        manager.postMessage({ type: 'needHop', fillLevel: currentFill() });
      }
      return;
    }
  };

  // UI wiring
  wireUI();
}

// ------------------ UI wiring ------------------
function wireUI() {
  playBtn?.addEventListener('click', async () => {
    if (isPlaying) return;

    // iOS/Chrome autoplay guard
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }

    isPlaying = true;

    // Arm the worklet
    workletNode.parameters.get('active').setValueAtTime(1, audioCtx.currentTime);

    // Prime one hop immediately (manager will queue it if not fully ready yet)
    manager.postMessage({ type: 'needHop', fillLevel: currentFill() });
  });

  stopBtn?.addEventListener('click', () => {
    if (!isPlaying) return;
    isPlaying = false;

    // Disarm the worklet
    workletNode.parameters.get('active').setValueAtTime(0, audioCtx.currentTime);
  });

  // Slider only updates the label; conditioning is sent with each needHop
  fillSlider?.addEventListener('input', () => {
    updateFillLabel();
    // No immediate message here: we send the current FillLevel
    // bundled with each 'needHop' for the freshest conditioning.
  });
}

// Kick things off after user gesture (safest)
window.addEventListener('DOMContentLoaded', () => {
  // // Optional: gate boot behind first user click to satisfy strict autoplay policies
  // const firstTap = () => {
  //   window.removeEventListener('click', firstTap);
  //   boot().catch(err => console.error('Boot error:', err));
  // };
  // window.addEventListener('click', firstTap, { once: true });

  // Or, if you prefer to boot immediately (and rely on Play button to resume):
  boot().catch(err => console.error('Boot error:', err));
});
