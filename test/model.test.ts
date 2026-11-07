import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseLlama2c, parseWspk, seedModel, serializeWspk, toLlama2c } from '../src/model/formats.ts';
import { gaussian } from '../src/model/random.ts';
import { BOS, Tokenizer } from '../src/model/tokenizer.ts';
import { parameterCount, tensorSpecs } from '../src/model/types.ts';
import { tinyTokenizer } from './helpers.ts';

test('BPE merges by score, with a dummy prefix and byte fallback', () => {
  const tok = tinyTokenizer();
  const ids = tok.encode('hello cat');
  assert.deepEqual(ids.map((i) => tok.pieces[i]), ['<s>', ' hello', ' cat']);
  assert.equal(tok.decode(ids), 'hello cat');
  const emoji = tok.encode('é');
  assert.ok(emoji.slice(2).every((i) => /^<0x/.test(tok.pieces[i])), 'unknown characters fall back to bytes');
  assert.equal(tok.decode(emoji), 'é');
});

test('tokenizer round-trips through the llama2.c file format', () => {
  const tok = tinyTokenizer();
  const back = Tokenizer.fromLlama2c(tok.toLlama2c(), tok.vocabSize);
  assert.deepEqual(back.pieces, tok.pieces);
  assert.deepEqual([...back.scores], [...tok.scores]);
  assert.deepEqual(back.encode('hello cat'), tok.encode('hello cat'));
});

test('byte-level tokenizer encodes any text', () => {
  const tok = Tokenizer.byteLevel();
  const ids = tok.encode('Hi ✓');
  assert.equal(ids[0], BOS);
  assert.equal(tok.decode(ids), 'Hi ✓');
});

test('llama2.c checkpoints round-trip, shared and separate classifier', () => {
  for (const shared of [true, false]) {
    const seed = seedModel({ dim: 8, layers: 2, heads: 2, kvHeads: 1, hidden: 12, seqLen: 16, vocabSize: 20 });
    seed.config.sharedClassifier = shared;
    if (!shared) seed.weights.set('lm_head', new Float32Array(20 * 8).map((_, i) => i / 100));
    const parsed = parseLlama2c(toLlama2c(seed));
    assert.equal(parsed.config.sharedClassifier, shared);
    assert.equal(parsed.config.headSize, 4);
    assert.deepEqual(parsed.config.layers[0].headKv, [0, 0]);
    for (const spec of tensorSpecs(seed.config)) {
      assert.deepEqual([...parsed.weights.get(spec.name)!], [...seed.weights.get(spec.name)!], spec.name);
    }
  }
});

test('llama2.c header garbage is rejected with a clear error', () => {
  assert.throws(() => parseLlama2c(new Int32Array([1, 2, 3, 4, 5, 0, 7]).buffer), /Not a llama2.c checkpoint/);
});

test('wspk keeps per-layer shapes', () => {
  const seed = seedModel({ dim: 8, layers: 2, heads: 2, kvHeads: 2, hidden: 4, seqLen: 16 });
  seed.config.layers[1] = { nHeads: 3, headKv: [0, 1, 1], hidden: 7 };
  for (const s of tensorSpecs(seed.config)) {
    if (s.name.startsWith('layer_1.')) seed.weights.set(s.name, new Float32Array(s.shape[0] * s.shape[1]).fill(0.5));
  }
  const back = parseWspk(serializeWspk(seed.config, seed.weights, { modelId: 'm', createdAt: 'now' }));
  assert.deepEqual(back.config, seed.config);
  assert.equal(back.meta.modelId, 'm');
  assert.equal(back.weights.get('layer_1.mlp.down')!.length, 8 * 7);
  assert.equal(parameterCount(back.config), parameterCount(seed.config));
});

test('a zero-capacity seed has no attention or MLP parameters', () => {
  const { config } = seedModel({ heads: 0, hidden: 0, dim: 32, layers: 1, kvHeads: 1 });
  const layerParams = tensorSpecs(config)
    .filter((s) => s.name.includes('.attn.wq') || s.name.includes('.mlp.'))
    .reduce((n, s) => n + s.shape[0] * s.shape[1], 0);
  assert.equal(layerParams, 0);
});

test('gaussian noise is standard normal and reproducible', () => {
  let sum = 0;
  let sq = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) {
    const g = gaussian(3, i);
    sum += g;
    sq += g * g;
  }
  assert.ok(Math.abs(sum / n) < 0.03);
  assert.ok(Math.abs(sq / n - 1) < 0.05);
  assert.equal(gaussian(3, 17), gaussian(3, 17));
  assert.notEqual(gaussian(3, 17), gaussian(4, 17));
});
