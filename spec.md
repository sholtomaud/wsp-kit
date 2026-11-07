wsp-kit: Browser-Native Weight-Space Programming (PoC)

Vision

wsp-kit is a TypeScript-based framework designed to facilitate Weight-Space Programming (WSP) entirely within the browser. It allows frontier LLMs (acting via API as the "Chief Architect") to iteratively read, mutate, prune, and grow a local neural network running in WebGPU.

The ultimate goal is to distill gigabyte-scale models into hyper-specialized, megabyte-scale Fodorian modules, or to grow task-specific networks from a zero-byte seed, overcoming traditional backpropagation and hardware constraints.

System Architecture

The system utilizes a "Hierarchical Optimization" model, divided into three distinct layers:

1. The Human Layer (tsrs-monaco)

Role: The user interface.

UX Philosophy: Natural language only (the "Hex" approach). The human developer does not write JSON or structured prompts. They simply type commands like: "The model fails at Python indents. Fix the attention heads responsible for whitespace."

2. The Harness Layer (wsp-kit)

Role: The orchestrator and translation bridge.

Function: Captures the human's natural language, invisibly attaches strict tool-calling schemas, and sends the payload to the external LLM API (Claude/Gemini). When the LLM responds with a structured JSON action, wsp-kit parses it and delegates the heavy lifting to the local WebGPU environment.

Constraint Management: Prevents the LLM from attempting to read or write gigabytes of raw floats. It forces the LLM to use high-level abstraction tools.

3. The Vat (fluxkit + WebGPU)

Role: The execution engine.

Function: Hosts the raw Float32Array buffers. Executes fast compute shaders for targeted weight mutations, procedural initializations, and localized micro-training loops (the "Intern Agents") without moving massive amounts of data back to the CPU or over the network.

Core Subsystems to Build

1. OPFS State Manager (The Safety Net)

LLMs will make catastrophic mistakes when editing weights. Instant state rollback is mandatory.

Requirement: Bind wsp-kit to the browser's Origin Private File System (OPFS).

Methods Required:

wsp.state.checkpoint(id: string): Instantly serializes current WebGPU VRAM state to local disk.

wsp.state.restore(id: string): Re-loads binary blobs from OPFS into WebGPU buffers.

2. Context Window Manager (Read Ergonomics)

The LLM cannot read 8 billion parameters. It needs semantic summaries and targeted slices.

Requirement: Utility functions to map WebGPU buffers and format them into LLM-readable text.

Methods Required:

wsp.inspect.summary(layerTarget): Returns dimensions, sparsity, mean, and variance.

wsp.inspect.slice(layerTarget, bounds): Returns a Markdown-formatted grid of specific coordinates.

wsp.inspect.findAnomalies(layerTarget, threshold): Uses a WebGPU shader to find mathematically relevant neurons and returns their coordinates.

3. The "Intern" Delegation API (Write & Grow Ergonomics)

Instead of the LLM passing arrays of floats, it passes high-level procedural commands via structured JSON outputs. wsp-kit translates these into fluxkit compute shaders.

Requirement: Strict JSON schemas (tool definitions) provided to the LLM.

Methods/Tools Required:

wsp.mutate.graftNode(config): Procedurally adds capacity (e.g., a new attention head) and initializes it using WebGPU shaders based on LLM heuristics.

wsp.mutate.patchSparse(layerTarget, edits[]): Takes a JSON array of [row, col, value] and dispatches a compute shader to overwrite only those specific floats in VRAM.

wsp.mutate.pruneSemantic(triggerDataset, threshold): Evaluates a dataset locally, tracks activation, and zeroes out the most active weights to remove specific domain knowledge.

wsp.mutate.microTrain(targetModule, dataset, steps): Freezes the rest of the network and runs a lightning-fast gradient update purely on a specific sub-circuit in WebGPU.

4. The Eval-Grow Loop

The feedback mechanism for the LLM to verify its Weight-Space Programming.

Requirement: A method for the LLM to run isolated forward passes.

Methods Required:

wsp.eval.run(prompt, expectedOutput): Runs inference locally in WebGPU and returns the loss, perplexity, or generated text to the LLM so it can decide whether to keep or rollback its last mutation.

Implementation Roadmap for Claude

When starting this project, please follow this build order:

Scaffold the Harness: Set up the basic TypeScript class structure for wsp-kit and the outbound API bridge to the LLM provider, utilizing strict Tool Schemas (function calling).

Bind OPFS: Implement the high-speed binary read/write to the browser's local file system.

Integrate fluxkit: Create the bridge that maps string identifiers (e.g., "layer_3.mlp.gate") to actual WebGPU memory pointers.

Write the Mutation Shaders: Write the WGSL (WebGPU Shading Language) scripts necessary to execute the sparse patches and procedural initializations locally.

Wire the UX: Connect the wsp-kit harness to the tsrs-monaco editor so natural language inputs trigger the orchestration loop.
