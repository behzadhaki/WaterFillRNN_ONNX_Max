# WaterFillRNN ONNX for the Browser

## Overview

This project implements a real-time neural audio generation system using RNN-based codec synthesis with Web Audio API worklets.

## File Structure

```
├── artifacts [precomputed Encodec token-> latents table]
│   ├── encodec24_codebooks.f16bin
│   └── encodec24_codebooks.meta.json
├── css
│   └── style.css
├── index.html
├── libs
│   ├── ort.wasm.min.mjs
│   ├── ort-wasm-simd-threaded.mjs
│   └── ort-wasm-simd-threaded.wasm
├── main.js   [js code for handeling UI events, firing up workers]
├── onnx   
│   ├── encodec_decode.onnx  [24kHz Encodec docoder only]
│   └── rnn_step.onnx        [RNN model that computes a single prediction step]
├── utils
│   └── resampler.js         [The browser runs at 48kHz, so we upsample after decoding]
├── workers
│   ├── manager-worker.js    [manages communication between main thread and RNN worker.]
│   └── RNNWorker.js
└── worklets
    └── WaterFillRNNWorklet.js   [The WebAudio AudioWorklet node, process() returns a single b128-sample buffer of audio]
```
## How it works

Main.js is the module that coordinates beteen the HTML UI events and the WebAudio graph it creates. The WebAudio graph is nothing but the AudioWorklet fronting for the NN and connecting to a Gain n ode for output. The AudioWorklet has to service the WebAudio calls with buffers of length 128 representing samples at 44kHz. It keeps a small "ring buffer" of audio around from the backend that it draws from. When the ring buffer get low, it asks the backend to generate another few "hops".

The "backend" consists of a manager-worker, and the workder that wraps the RNN. The RNN just takes one latent vector input and its own hidden state from the previous step, and generates a single step.
The manager responds to "getHop" calls from the Worklets. It calls the RNN worker for one step at a time, get the logits for 8 codebooks, samples them to choose the codes, then does two things with the codes: 1) decodes them to latents to feed the RNN for the next step, and 2) saves the codes in a FIFO buffer (which is longer than the number of steps is is producing for a hop. - Why? We need more than a single hop (8 steps) for the Encodec decoder to work well - so we keep a short history of hops (a "chunk") in the buffer. After 8 steps (a "hop") is computed and stuffed on the FIFO chunk, we decode the chunk, and since we get more audio than we need, we chop off the end of the audio returned so it is equivalent in lenth to what the Worklet asked for and return it to the worklet. 

The ONNX Decoder is used only by the manager to create the audio from the chunk of codes.


## How to Run

```bash
# Start a local web server in the project root directory
python3 -m http.server 8000

# Then open your browser to:
# http://localhost:8000
```

**Note:** This project requires a local web server due to ES6 module imports and Web Workers. Opening `index.html` directly in a browser will not work.
