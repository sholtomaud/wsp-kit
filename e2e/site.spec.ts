/**
 * The site end to end: a starter model "downloaded" (Hugging Face is mocked
 * with a synthetic llama2.c checkpoint), a key entered, and one architect turn
 * against a mocked Anthropic API that calls real tools on the WebGPU network.
 */
import { expect, test, type Page } from '@playwright/test';
import { seedModel, toLlama2c } from '../src/model/formats.ts';
import { Tokenizer } from '../src/model/tokenizer.ts';

function syntheticStarter() {
  const byte = Tokenizer.byteLevel();
  const pieces = [...byte.pieces];
  for (let i = 0; pieces.length < 512; i++) pieces.push(`w${i}`);
  const tok = new Tokenizer(pieces, new Float32Array(512));
  const model = seedModel({ dim: 32, layers: 2, heads: 4, kvHeads: 2, hidden: 48, seqLen: 128, vocabSize: 512, std: 0.2, seed: 3 });
  return { model: Buffer.from(toLlama2c(model)), tokenizer: Buffer.from(tok.toLlama2c()) };
}

async function mockHuggingFace(page: Page) {
  const files = syntheticStarter();
  let downloads = 0;
  await page.route('https://huggingface.co/**', (route) => {
    downloads++;
    const url = route.request().url();
    return route.fulfill({ status: 200, body: url.endsWith('tok512.bin') ? files.tokenizer : files.model, headers: { 'access-control-allow-origin': '*' } });
  });
  return () => downloads;
}

async function loadStarter(page: Page) {
  await page.selectOption('#source', 'stories260K');
  await page.click('#load');
  await expect(page.locator('#load-status')).toContainText('Loaded stories260K', { timeout: 30_000 });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.evaluate(async () => {
    localStorage.clear();
    sessionStorage.clear();
    const root = await navigator.storage.getDirectory();
    await root.removeEntry('wsp-kit', { recursive: true }).catch(() => {});
  });
  await page.reload();
});

test('a starter loads onto WebGPU, is cached in OPFS, and generates text', async ({ page }) => {
  const downloads = await mockHuggingFace(page);
  await loadStarter(page);
  await expect(page.locator('#backend-badge')).toContainText('WebGPU');
  expect(downloads()).toBe(2);
  await page.click('#load');
  await expect(page.locator('#load-status')).not.toContainText('Loaded');
  await expect(page.locator('#load-status')).toContainText('Loaded stories260K');
  expect(downloads()).toBe(2); // second load came from OPFS
  await page.locator('.playground summary').click();
  await page.click('#play-run');
  await expect(page.locator('#play-out')).toContainText('tokens', { timeout: 30_000 });
});

test('a seed network can be created from zero capacity', async ({ page }) => {
  await page.selectOption('#source', 'seed');
  await page.fill('#seed-heads', '0');
  await page.fill('#seed-hidden', '0');
  await page.click('#load');
  await expect(page.locator('#load-status')).toContainText('Loaded seed-');
  await expect(page.locator('#model-badge')).toContainText('params');
});

test('the API key is forgotten on refresh by default', async ({ page }) => {
  await page.selectOption('#provider', 'anthropic');
  await page.fill('#api-key', 'sk-ant-test-123');
  await page.click('#key-save');
  await expect(page.locator('#key-status')).toContainText('forgotten on refresh');
  const stored = await page.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }));
  expect(stored).not.toContain('sk-ant-test-123');
  await page.reload();
  await expect(page.locator('#key-status')).toContainText('No key set');
});

test('"Keep for this tab" survives a refresh', async ({ page }) => {
  await page.selectOption('#key-persist', 'session');
  await page.fill('#api-key', 'sk-ant-tab');
  await page.click('#key-save');
  await page.reload();
  await expect(page.locator('#key-status')).toContainText('kept for this tab');
});

test('an architect turn: Claude calls tools, the harness checkpoints, the transcript shows it all', async ({ page }) => {
  await mockHuggingFace(page);
  const requests: { body: any; headers: Record<string, string> }[] = [];
  const replies = [
    {
      content: [
        { type: 'text', text: 'I will look, then patch one value.' },
        { type: 'tool_use', id: 'tu_1', name: 'model_overview', input: {} },
        { type: 'tool_use', id: 'tu_2', name: 'mutate_patch_sparse', input: { target: 'layer_0.mlp.gate', edits: [{ row: 0, col: 0, value: 0.5 }] } },
      ],
      stop_reason: 'tool_use',
    },
    { content: [{ type: 'tool_use', id: 'tu_3', name: 'eval_run', input: { prompt: 'Once upon a time', expected_output: ' there', max_tokens: 4 } }], stop_reason: 'tool_use' },
    { content: [{ type: 'text', text: 'Patched layer_0.mlp.gate[0,0]; the loss is reported above.' }], stop_reason: 'end_turn' },
  ];
  await page.route('https://api.anthropic.com/**', async (route) => {
    if (route.request().method() === 'OPTIONS') {
      return route.fulfill({ status: 204, headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-allow-methods': 'POST' } });
    }
    requests.push({ body: route.request().postDataJSON(), headers: route.request().headers() });
    const r = replies[requests.length - 1];
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: { 'access-control-allow-origin': '*' },
      body: JSON.stringify({ id: `msg_${requests.length}`, type: 'message', role: 'assistant', model: 'claude-opus-5-5', usage: { input_tokens: 100, output_tokens: 20 }, ...r }),
    });
  });

  await loadStarter(page);
  await page.fill('#api-key', 'sk-ant-test');
  await page.click('#key-save');
  await page.fill('#message', 'Look at the network and nudge one MLP weight.');
  await page.keyboard.press('Enter');

  await expect(page.locator('.msg.assistant').last()).toContainText('Patched layer_0.mlp.gate', { timeout: 30_000 });
  await expect(page.locator('.tool .name')).toHaveText(['model_overview', 'mutate_patch_sparse', 'eval_run']);
  await expect(page.locator('.tool.err')).toHaveCount(0);
  await expect(page.locator('.tool').nth(1).locator('.meta')).toContainText('saved auto-1');
  await expect(page.locator('#ckpt-list')).toContainText('auto-1');

  expect(requests).toHaveLength(3);
  expect(requests[0].headers['anthropic-dangerous-direct-browser-access']).toBe('true');
  expect(requests[0].body.model).toBe('claude-opus-5-5');
  expect(requests[0].body.tools.every((t: any) => t.strict === true)).toBe(true);
  // The second request returns both tool results in one user turn.
  const results = requests[1].body.messages.at(-1).content;
  expect(results.map((b: any) => b.type)).toEqual(['tool_result', 'tool_result']);
  expect(JSON.parse(results[0].content).tensors.length).toBe(1 + 2 * 9 + 1);

  // The patch really happened in VRAM, and the auto checkpoint undoes it.
  const values = await page.evaluate(async () => {
    const w = (globalThis as any).wsp;
    const now = (await w.backend.read('layer_0.mlp.gate', 0, 1))[0];
    await w.state.restore('auto-1');
    const restored = (await w.backend.read('layer_0.mlp.gate', 0, 1))[0];
    return { now, restored };
  });
  expect(values.now).toBe(0.5);
  expect(values.restored).not.toBe(0.5);
});

test('provider errors are shown, not swallowed', async ({ page }) => {
  await mockHuggingFace(page);
  await page.route('https://api.anthropic.com/**', (route) =>
    route.fulfill({
      status: 401,
      contentType: 'application/json',
      headers: { 'access-control-allow-origin': '*' },
      body: JSON.stringify({ type: 'error', error: { type: 'authentication_error', message: 'invalid x-api-key' } }),
    })
  );
  await loadStarter(page);
  await page.fill('#api-key', 'bad');
  await page.click('#key-save');
  await page.fill('#message', 'hello');
  await page.click('#send');
  await expect(page.locator('.msg.error')).toContainText('rejected the API key');
});

test('the page fits a phone screen', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 740 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test('each provider shows only its own settings', async ({ page }) => {
  const visible = async () => ({
    url: await page.locator('#base-url-row').isVisible(),
    effort: await page.locator('#effort-row').isVisible(),
    openrouter: await page.locator('#openrouter-free').isVisible(),
  });
  await page.selectOption('#provider', 'anthropic');
  expect(await visible()).toEqual({ url: false, effort: true, openrouter: false });
  await page.selectOption('#provider', 'openrouter');
  expect(await visible()).toEqual({ url: false, effort: false, openrouter: true });
  await page.selectOption('#provider', 'llamacpp');
  expect(await visible()).toEqual({ url: true, effort: false, openrouter: false });
  await expect(page.locator('#key-status')).toContainText('No key needed');
});
