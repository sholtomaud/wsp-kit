# wsp-kit

Browser-native **Weight-Space Programming**. A small transformer runs in your browser on WebGPU. A frontier LLM (the "Chief Architect") reads and edits its weights directly through tools: it inspects, patches, grafts, prunes and evaluates. It works on the network by editing weights instead of training it.

You type plain language ("the model keeps saying *Lily*; find out why and remove it without hurting other stories"). wsp-kit attaches the tool schemas, sends the request to the LLM, runs each tool call as a WebGPU compute shader, and returns small, readable results. A checkpoint is saved before every edit so the architect can roll back.

See [spec.md](spec.md) for the original design.

## Run it

```bash
npm install
```

```bash
npm run dev
```

Open the printed URL in **Chrome, Edge or Safari** (WebGPU; Firefox is not targeted). Then:

1. **Network**: pick *TinyStories 260K* (loads in a second) and press *Load network*. Starter models are downloaded once from Hugging Face and then kept in the browser's private file system (OPFS).
2. **Architect**: choose a provider, paste a key, press *Use key*.
3. Type what you want.

The *Playground* at the bottom runs the network directly, without an LLM.

## Providers and keys

| Provider | Key | Notes |
| --- | --- | --- |
| Anthropic (default `claude-opus-5-5`) | yes | Official SDK, called straight from the browser. Strict tool schemas, adaptive thinking, selectable effort, and server-side refusal fallback on models that support it. |
| OpenRouter | yes | *List OpenRouter models with tool support* fetches the catalogue (free models first). |
| Google Gemini | yes | `generateContent` with function declarations; thought signatures are replayed. |
| llama.cpp (local) | optional | Run `llama-server -m model.gguf --jinja` (tool calling needs `--jinja`). Chrome asks for permission before an HTTPS page can reach `http://localhost`. Safari blocks it, so with Safari run the site locally (`npm run dev`). |

**Keys never leave the browser except to their own provider.** By default a key is kept **in memory only and forgotten on refresh**. You can choose to keep it for the tab (sessionStorage) or on the device (localStorage). Keys are never written into checkpoints, exports or the conversation. There is no server and no shared key, and nothing needs to be configured in GitHub secrets.

## Hosting (GitHub Pages)

`.github/workflows/deploy.yml` runs on every push to `main`. It typechecks, runs the unit tests and the browser tests (WebGPU on SwiftShader inside the Playwright container), builds with Vite, and deploys `dist/` to Pages. Pages must be set to **Settings → Pages → Source: GitHub Actions**. `ci.yml` runs the same checks on pull requests and branches. The build uses relative URLs, so it works at `https://<user>.github.io/wsp-kit/`.

## How the spec maps to the code

| Spec | Code |
| --- | --- |
| Harness layer: NL in, strict tool schemas, LLM API bridge | `src/harness/agent.ts` (loop, auto-checkpoints), `src/harness/tools.ts` (15 tools), `src/harness/schema.ts` (validation), `src/llm/*` (providers) |
| The Vat: Float32 buffers in WebGPU, compute shaders | `src/gpu/backend.ts`, `src/gpu/kernels.ts` (WGSL) |
| `wsp.state.checkpoint/restore` (OPFS) | `src/wsp.ts`, `src/state/store.ts`; `.wspk` format in `src/model/formats.ts` |
| `wsp.inspect.summary/slice/findAnomalies` | `src/wsp.ts`, plus `inspect.neuron` (logit lens) and `inspect.tokenize` |
| `wsp.mutate.patchSparse/graftNode/pruneSemantic/microTrain` | `src/wsp.ts` |
| `wsp.eval.run` | `src/wsp.ts` (greedy generation, loss, perplexity, delta since the last eval), plus `eval.nextTokens` |
| String ids → GPU buffers (`"layer_3.mlp.gate"`) | `src/model/types.ts` (tensor naming), `Backend` in `src/backend.ts` |

### Design notes

- **Model.** A Llama-style decoder (RMSNorm, rotary attention with grouped KV heads, SwiGLU MLP), compatible with Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) checkpoints. Unlike llama2.c, **each layer keeps its own width**, and every query head names the KV head it reads, so heads and neurons can be grafted onto one layer.
- **Every edit runs on the GPU.** Scatter, fill (seeded Gaussian, identical in WGSL and TypeScript), region copy, statistics, outlier search and row norms are all shaders. Only what the LLM asks to see is read back: statistics, a grid of at most 32×16 values, ranked coordinates, logits.
- **No backpropagation.** `microTrain` is a gradient-free (1+1) evolution strategy: it adds seeded noise to one tensor, keeps the change only if the loss falls, and leaves the rest of the network frozen. The architect's main levers are direct edits.
- **Function-preserving growth.** `graftNode` defaults to zero output weights, so a new neuron or head changes nothing until it is edited. `init: "clone"` with `output: "split"` is Net2Net: it duplicates a neuron and shares its output.
- **Semantic pruning works per neuron.** It tracks each MLP neuron's mean |activation| on the trigger texts (optionally against contrast texts to keep), then zeroes the most specific neurons' gate and up rows and down column.
- **CPU reference.** Every operation also exists as plain TypeScript. The browser tests compare WebGPU against it: logits agree to within 1e-3, and greedy generations are identical.

## Use it as a library

```ts
import { createWsp, loadStarter, Harness, AnthropicProvider } from 'wsp-kit';

const wsp = await createWsp(await loadStarter('stories15M'));
await wsp.inspect.summary('layer_3.mlp.gate');
await wsp.inspect.findAnomalies('layer_3.mlp.gate', 4, { mode: 'rows' });
await wsp.state.checkpoint('before');
await wsp.mutate.graftNode({ layer: 3, kind: 'mlp_neurons', count: 16 });
await wsp.eval.run('Once upon a time', ' there was');

const harness = new Harness({ wsp, provider: () => new AnthropicProvider({ apiKey }), onEvent: console.log });
await harness.send('Make the model less likely to name every character Lily.');
```

On the site, the loaded instance is also available as `window.wsp` in the console.

## Tests

```bash
npm test
```

```bash
npm run e2e
```

`npm test` runs the Node unit tests: model formats, tokenizer, every `wsp.*` method on the CPU backend, the agent loop, and each provider's wire format. `npm run e2e` runs Playwright in Chromium with WebGPU on SwiftShader: GPU-vs-CPU equivalence for every kernel, OPFS checkpoints, and the full site with mocked Hugging Face and Anthropic endpoints.

## Not done yet

- **tsrs-monaco integration** (spec step 5). The harness is a plain class (`Harness`) with an event stream, ready to embed. The wiring itself belongs in tsrs-monaco.
- **Larger models.** Weights are f32 and a tensor must fit in one storage buffer (`maxStorageBufferBindingSize`, often 128 MB–2 GB). The TinyStories models (260K to 42M) fit easily. Gigabyte-scale models need f16 or quantized weights and sharded tensors.
- **Forward-pass speed.** It runs one token per submit, with one thread per output row in the matrix-vector kernel. That is fine for these sizes, but tiled kernels would be the next step.
- Attention-head pruning, and a column-norm mode for `findAnomalies` (MLP neurons are columns of `mlp.down`).
