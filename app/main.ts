/**
 * The wsp-kit site: load a network into WebGPU, connect an LLM, and talk to it.
 */
import { Harness, type AgentEvent } from '../src/harness/agent.ts';
import { AnthropicProvider, ANTHROPIC_DEFAULT_MODEL, type Effort } from '../src/llm/anthropic.ts';
import { GeminiProvider, GEMINI_DEFAULT_MODEL } from '../src/llm/gemini.ts';
import { keys, type KeyPersistence } from '../src/llm/keys.ts';
import { LLAMACPP_BASE, listOpenRouterModels, OpenAiCompatibleProvider, OPENROUTER_BASE } from '../src/llm/openai-compatible.ts';
import type { Provider } from '../src/llm/types.ts';
import { createWsp, loadFile, loadSeed, loadStarter, STARTERS, type LoadedModel } from '../src/loader.ts';
import { parameterCount } from '../src/model/types.ts';
import type { WSP } from '../src/wsp.ts';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

type ProviderId = 'anthropic' | 'openrouter' | 'gemini' | 'llamacpp';

const PROVIDERS: Record<ProviderId, { model: string; needsKey: boolean; models: string[] }> = {
  anthropic: { model: ANTHROPIC_DEFAULT_MODEL, needsKey: true, models: ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-fable-5-1'] },
  openrouter: { model: 'openrouter/auto', needsKey: true, models: [] },
  gemini: { model: GEMINI_DEFAULT_MODEL, needsKey: true, models: ['gemini-2.5-flash', 'gemini-2.5-pro'] },
  llamacpp: { model: 'local', needsKey: false, models: [] },
};

// Non-secret preferences (provider, model, server URL) survive reloads; keys follow their own setting.
const prefs = {
  get(name: string, fallback = '') {
    try {
      return localStorage.getItem(`wsp-kit:pref:${name}`) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(name: string, value: string) {
    try {
      localStorage.setItem(`wsp-kit:pref:${name}`, value);
    } catch {
      /* storage blocked */
    }
  },
};

let wsp: WSP | undefined;
let harness: Harness | undefined;
let abort: AbortController | undefined;

// ------------------------------------------------------------- network

const source = $<HTMLSelectElement>('source');
for (const s of STARTERS) source.add(new Option(`${s.label} (~${formatBytes(s.approxBytes)})`, s.id));
source.add(new Option('Seed: grow a network from scratch', 'seed'));
source.add(new Option('Open files from this computer', 'file'));
source.value = prefs.get('source', 'stories260K');

function showSource() {
  const id = source.value;
  $('seed-options').hidden = id !== 'seed';
  $('file-options').hidden = id !== 'file';
  const s = STARTERS.find((m) => m.id === id);
  $('source-desc').textContent = s
    ? `${s.description} Downloaded once from Hugging Face, then kept in this browser's private storage.`
    : id === 'seed'
      ? 'A fresh network with a byte-level vocabulary.'
      : 'A llama2.c checkpoint with its tokenizer.bin, or a .wspk file exported from wsp-kit.';
  prefs.set('source', id);
}
source.addEventListener('change', showSource);
showSource();

function setStatus(el: HTMLElement, text: string, kind: '' | 'error' | 'ok' = '') {
  el.textContent = text;
  el.className = `hint ${kind}`;
}

async function readFile(input: HTMLInputElement): Promise<{ data: ArrayBuffer; name: string } | undefined> {
  const f = input.files?.[0];
  return f ? { data: await f.arrayBuffer(), name: f.name } : undefined;
}

$('load').addEventListener('click', async () => {
  const button = $<HTMLButtonElement>('load');
  const status = $('load-status');
  const progress = $<HTMLProgressElement>('load-progress');
  button.disabled = true;
  setStatus(status, 'Loading…');
  progress.hidden = false;
  progress.removeAttribute('value');
  try {
    let loaded: LoadedModel;
    if (source.value === 'seed') {
      const num = (id: string) => Number($<HTMLInputElement>(id).value);
      loaded = loadSeed({ dim: num('seed-dim'), layers: num('seed-layers'), heads: num('seed-heads'), kvHeads: Math.max(1, num('seed-heads')), hidden: num('seed-hidden'), seed: Date.now() % 100000 });
    } else if (source.value === 'file') {
      const model = await readFile($<HTMLInputElement>('file-model'));
      if (!model) throw new Error('Choose a model file first');
      const tok = await readFile($<HTMLInputElement>('file-tokenizer'));
      loaded = loadFile(model.data, tok?.data, model.name);
    } else {
      const totals = new Map<string, { loaded: number; total: number }>();
      loaded = await loadStarter(source.value, (p) => {
        totals.set(p.file, p);
        const all = [...totals.values()];
        const total = all.reduce((n, x) => n + x.total, 0);
        if (total) progress.value = all.reduce((n, x) => n + x.loaded, 0) / total;
        setStatus(status, p.cached ? 'Loading from this browser…' : `Downloading ${formatBytes(all.reduce((n, x) => n + x.loaded, 0))}…`);
      });
    }
    setStatus(status, 'Uploading weights to the GPU…');
    const next = await createWsp(loaded, { backend: $<HTMLInputElement>('use-cpu').checked ? 'cpu' : 'auto' });
    if (harness?.running) abort?.abort();
    const previous = wsp;
    wsp = next;
    previous?.model.backend.destroy?.();
    (globalThis as { wsp?: WSP }).wsp = wsp;
    harness = new Harness({ wsp, provider: makeProvider, onEvent: render });
    clearTranscript();
    updateBadges();
    const c = wsp.config;
    setStatus(status, `Loaded ${loaded.id}: ${formatCount(parameterCount(c))} parameters, ${c.layers.length} layers, vocabulary ${c.vocabSize}.`, 'ok');
    await refreshCheckpoints();
  } catch (e) {
    setStatus(status, (e as Error).message, 'error');
    console.error(e);
  } finally {
    button.disabled = false;
    progress.hidden = true;
  }
});

function updateBadges() {
  const b = $('backend-badge');
  const m = $('model-badge');
  if (!wsp) return;
  b.textContent = wsp.backend.description;
  b.classList.toggle('on', wsp.backend.kind === 'webgpu');
  m.textContent = `${wsp.model.id} · ${formatCount(parameterCount(wsp.config))} params`;
  m.classList.add('on');
  $('store-kind').textContent = wsp.store.kind === 'opfs' ? '(browser file system)' : '(memory only)';
}

// ------------------------------------------------------------- provider

const providerSel = $<HTMLSelectElement>('provider');
const modelInput = $<HTMLInputElement>('llm-model');
const keyInput = $<HTMLInputElement>('api-key');
const persistSel = $<HTMLSelectElement>('key-persist');
const baseUrl = $<HTMLInputElement>('base-url');
providerSel.value = prefs.get('provider', 'anthropic');
persistSel.value = prefs.get('key-persist', 'memory');

function showProvider() {
  const id = providerSel.value as ProviderId;
  const p = PROVIDERS[id];
  prefs.set('provider', id);
  modelInput.value = prefs.get(`model:${id}`, p.model);
  const list = $('llm-models');
  list.replaceChildren(...p.models.map((m) => new Option(m, m)));
  $('openrouter-free').hidden = id !== 'openrouter';
  $('base-url-row').hidden = id !== 'llamacpp';
  $('effort-row').hidden = id !== 'anthropic';
  baseUrl.value = prefs.get('llamacpp-url', LLAMACPP_BASE);
  keyInput.value = '';
  keyInput.placeholder = p.needsKey ? 'Paste your key' : 'Optional (llama-server --api-key)';
  showKeyStatus();
}

function showKeyStatus() {
  const id = providerSel.value;
  const where = keys.where(id);
  const status = $('key-status');
  if (!where) setStatus(status, PROVIDERS[id as ProviderId].needsKey ? 'No key set.' : 'No key needed.');
  else setStatus(status, `Key set (${where === 'memory' ? 'forgotten on refresh' : where === 'session' ? 'kept for this tab' : 'kept on this device'}).`, 'ok');
}

providerSel.addEventListener('change', showProvider);
modelInput.addEventListener('change', () => prefs.set(`model:${providerSel.value}`, modelInput.value.trim()));
baseUrl.addEventListener('change', () => prefs.set('llamacpp-url', baseUrl.value.trim()));
persistSel.addEventListener('change', () => prefs.set('key-persist', persistSel.value));
$('key-save').addEventListener('click', () => {
  const key = keyInput.value.trim();
  if (!key) return setStatus($('key-status'), 'Paste a key first.', 'error');
  keys.set(providerSel.value, key, persistSel.value as KeyPersistence);
  keyInput.value = '';
  showKeyStatus();
});
keyInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') $('key-save').click();
});
$('key-clear').addEventListener('click', () => {
  keys.clear(providerSel.value);
  showKeyStatus();
});
$('openrouter-free').addEventListener('click', async () => {
  const button = $<HTMLButtonElement>('openrouter-free');
  button.disabled = true;
  try {
    const models = await listOpenRouterModels();
    $('llm-models').replaceChildren(...models.map((m) => new Option(`${m.free ? 'free · ' : ''}${m.name}`, m.id)));
    setStatus($('key-status'), `${models.filter((m) => m.free).length} free and ${models.filter((m) => !m.free).length} paid models support tools. Pick one in Model.`, 'ok');
  } catch (e) {
    setStatus($('key-status'), (e as Error).message, 'error');
  } finally {
    button.disabled = false;
  }
});
showProvider();

function makeProvider(): Provider {
  const id = providerSel.value as ProviderId;
  const model = modelInput.value.trim() || PROVIDERS[id].model;
  const key = keys.get(id);
  if (PROVIDERS[id].needsKey && !key) throw new Error(`Add your ${providerSel.selectedOptions[0].text} API key first (Architect panel).`);
  switch (id) {
    case 'anthropic':
      return new AnthropicProvider({ apiKey: key, model, effort: $<HTMLSelectElement>('effort').value as Effort });
    case 'gemini':
      return new GeminiProvider({ apiKey: key, model });
    case 'openrouter':
      return new OpenAiCompatibleProvider({
        id: 'openrouter',
        label: 'OpenRouter',
        baseUrl: OPENROUTER_BASE,
        apiKey: key,
        model,
        headers: { 'HTTP-Referer': location.origin, 'X-Title': 'wsp-kit' },
      });
    case 'llamacpp':
      return new OpenAiCompatibleProvider({ id: 'llamacpp', label: 'llama.cpp', baseUrl: baseUrl.value.trim() || LLAMACPP_BASE, apiKey: key || undefined, model });
  }
}

// ------------------------------------------------------------- conversation

const transcript = $('transcript');
const pendingTools = new Map<string, HTMLDetailsElement>();

function clearTranscript() {
  transcript.replaceChildren();
  pendingTools.clear();
  note(wsp ? `Network ready on ${wsp.backend.description}. Say what you want changed.` : 'Load a network to begin.');
}

function scrollDown() {
  transcript.scrollTop = transcript.scrollHeight;
}

function note(text: string, kind = 'notice') {
  const div = document.createElement('div');
  div.className = `msg ${kind}`;
  div.textContent = text;
  transcript.append(div);
  scrollDown();
}

/** Escaped text with `code` spans; nothing else is interpreted. */
function richText(el: HTMLElement, text: string) {
  text.split(/(`[^`\n]+`)/).forEach((part) => {
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      const code = document.createElement('code');
      code.textContent = part.slice(1, -1);
      el.append(code);
    } else el.append(document.createTextNode(part));
  });
}

function summarizeInput(input: unknown): string {
  const s = JSON.stringify(input ?? {});
  return s === '{}' ? '' : s.length > 120 ? s.slice(0, 117) + '…' : s;
}

function pretty(content: string): string {
  try {
    const v = JSON.parse(content);
    if (v && typeof v === 'object' && typeof v.markdown === 'string') {
      const { markdown, ...rest } = v;
      return `${JSON.stringify(rest, null, 2)}\n\n${markdown}`;
    }
    return JSON.stringify(v, null, 2);
  } catch {
    return content;
  }
}

function render(e: AgentEvent) {
  switch (e.type) {
    case 'user': {
      const div = document.createElement('div');
      div.className = 'msg user';
      div.textContent = e.text;
      transcript.append(div);
      break;
    }
    case 'assistant': {
      const div = document.createElement('div');
      div.className = 'msg assistant';
      if (e.model) {
        const m = document.createElement('span');
        m.className = 'model';
        m.textContent = e.model;
        div.append(m);
      }
      richText(div, e.text);
      transcript.append(div);
      break;
    }
    case 'tool_call': {
      const d = document.createElement('details');
      d.className = 'tool pending';
      d.innerHTML = '<summary><span class="name"></span><span class="meta"></span></summary><pre class="args"></pre><pre class="result"></pre>';
      d.querySelector('.name')!.textContent = e.call.name;
      d.querySelector('.meta')!.textContent = summarizeInput(e.call.input);
      d.querySelector('.args')!.textContent = JSON.stringify(e.call.input, null, 2);
      d.querySelector('.result')!.textContent = 'running…';
      pendingTools.set(e.call.id, d);
      transcript.append(d);
      break;
    }
    case 'tool_result': {
      const d = pendingTools.get(e.call.id);
      if (!d) break;
      d.classList.remove('pending');
      d.classList.toggle('err', e.result.isError);
      d.querySelector('.meta')!.textContent = `${e.ms} ms${e.checkpoint ? ` · saved ${e.checkpoint}` : ''}${e.result.isError ? ` · ${e.result.content.slice(0, 100)}` : ''} ${summarizeInput(e.call.input)}`;
      d.querySelector('.result')!.textContent = pretty(e.result.content);
      if (e.checkpoint) void refreshCheckpoints();
      break;
    }
    case 'stopped':
      note(e.reason);
      break;
    case 'error':
      note(e.message, 'msg error');
      break;
  }
  scrollDown();
}

const message = $<HTMLTextAreaElement>('message');
message.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $<HTMLFormElement>('composer').requestSubmit();
  }
});

$('composer').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = message.value.trim();
  if (!text) return;
  if (!wsp || !harness) return note('Load a network first (Network panel).', 'msg error');
  try {
    makeProvider();
  } catch (err) {
    return note((err as Error).message, 'msg error');
  }
  message.value = '';
  abort = new AbortController();
  $<HTMLButtonElement>('send').disabled = true;
  $('stop').hidden = false;
  try {
    await harness.send(text, abort.signal);
  } catch (err) {
    console.error(err);
  } finally {
    $<HTMLButtonElement>('send').disabled = false;
    $('stop').hidden = true;
    abort = undefined;
    void refreshCheckpoints();
  }
});
$('stop').addEventListener('click', () => abort?.abort());
$('clear-chat').addEventListener('click', () => {
  harness?.reset();
  clearTranscript();
});

// ------------------------------------------------------------- checkpoints

async function refreshCheckpoints() {
  const list = $('ckpt-list');
  if (!wsp) return list.replaceChildren();
  const items = await wsp.state.list();
  list.replaceChildren(
    ...items.map((c) => {
      const li = document.createElement('li');
      li.innerHTML = '<span class="name"></span><span class="size"></span><button class="small restore">Restore</button><button class="small ghost del" aria-label="Delete">✕</button>';
      li.querySelector('.name')!.textContent = c.id;
      (li.querySelector('.name') as HTMLElement).title = c.savedAt;
      li.querySelector('.size')!.textContent = formatBytes(c.bytes);
      li.querySelector('.restore')!.addEventListener('click', async () => {
        try {
          await wsp!.state.restore(c.id);
          updateBadges();
          note(`Restored checkpoint ${c.id}.`);
        } catch (e) {
          note((e as Error).message, 'msg error');
        }
      });
      li.querySelector('.del')!.addEventListener('click', async () => {
        await wsp!.state.delete(c.id);
        void refreshCheckpoints();
      });
      return li;
    })
  );
}

$('ckpt-save').addEventListener('click', async () => {
  if (!wsp) return;
  const input = $<HTMLInputElement>('ckpt-id');
  const id = input.value.trim() || `manual-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}`;
  try {
    await wsp.state.checkpoint(id, 'saved by hand');
    input.value = '';
    await refreshCheckpoints();
  } catch (e) {
    note((e as Error).message, 'msg error');
  }
});

$('export').addEventListener('click', async () => {
  if (!wsp) return;
  const data = await wsp.exportWspk('exported from the wsp-kit site');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([data], { type: 'application/octet-stream' }));
  a.download = `${wsp.model.id}.wspk`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ------------------------------------------------------------- playground

$('play-run').addEventListener('click', async () => {
  const out = $('play-out');
  if (!wsp) return (out.textContent = 'Load a network first.');
  const prompt = $<HTMLInputElement>('play-prompt').value;
  out.textContent = '…';
  try {
    const started = performance.now();
    const r = await wsp.model.generate(prompt, { maxTokens: 96, temperature: 0 });
    out.textContent = `${prompt}${r.text}\n\n(${r.tokens.length} tokens, ${Math.round(performance.now() - started)} ms, greedy)`;
  } catch (e) {
    out.textContent = (e as Error).message;
  }
});

// ------------------------------------------------------------- helpers

function formatBytes(n: number) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(0)} KB` : `${n} B`;
}
function formatCount(n: number) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);
}

if (!('gpu' in navigator)) {
  $('backend-badge').textContent = 'WebGPU unavailable: CPU only';
  $<HTMLInputElement>('use-cpu').checked = true;
}
