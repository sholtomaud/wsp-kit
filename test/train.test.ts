import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fullRegion, type Sequence } from '../src/backend.ts';
import { CpuBackend } from '../src/cpu/backend.ts';
import { seedModel } from '../src/model/formats.ts';
import { Model } from '../src/model/model.ts';
import { Tokenizer } from '../src/model/tokenizer.ts';
import { tensorSpecs } from '../src/model/types.ts';
import { makeWsp } from './helpers.ts';

/** GQA (4 query heads on 2 KV heads), a tied classifier, and one layer with its own widths. */
function gradModel() {
  const seed = seedModel({ dim: 16, layers: 2, heads: 4, kvHeads: 2, hidden: 12, seqLen: 32, std: 0.4, seed: 21 });
  seed.config.layers[1] = { nHeads: 3, headKv: [1, 0, 1], hidden: 7 };
  for (const s of tensorSpecs(seed.config)) {
    if (!s.name.startsWith('layer_1.')) continue;
    const n = s.shape[0] * s.shape[1];
    seed.weights.set(s.name, Float32Array.from({ length: n }, (_, i) => (s.name.endsWith('norm') ? 1 + 0.1 * Math.sin(i) : 0.4 * Math.sin(i * 1.7 + n))));
  }
  const tok = Tokenizer.byteLevel();
  const be = new CpuBackend();
  const model = new Model(be, tok, { ...seed.config, vocabSize: 259 }, seed.weights, 'g');
  return { be, model, tok };
}

const window = (tok: Tokenizer, text: string): Sequence => {
  const ids = tok.encode(text);
  return { tokens: ids.slice(0, -1), targets: ids.slice(1) };
};

test('CPU gradients match finite differences for every tensor', async () => {
  const { be, model, tok } = gradModel();
  const seqs = [window(tok, 'hello there!'), window(tok, 'abc')];
  seqs[1].targets[0] = -1; // an unscored position
  const targets = model.specs().map((s) => ({ name: s.name, region: fullRegion(s.shape) }));
  const trainer = be.createTrainer(model.config, 16, targets);
  await trainer.accumulate(seqs);
  const grads = await trainer.gradients();
  const meanLoss = async () => {
    const r = await trainer.loss(seqs);
    return r.sum / r.count;
  };
  let worst = 0;
  for (const spec of model.specs()) {
    const w = be.data(spec.name);
    const g = grads.get(spec.name)!;
    // The largest-gradient entry plus a few fixed ones.
    let big = 0;
    for (let i = 0; i < g.length; i++) if (Math.abs(g[i]) > Math.abs(g[big])) big = i;
    for (const i of new Set([big, 0, Math.floor(w.length / 2), w.length - 1])) {
      const orig = w[i];
      const eps = 1e-3;
      w[i] = orig + eps;
      const up = await meanLoss();
      w[i] = orig - eps;
      const down = await meanLoss();
      const actual = Math.fround(orig + eps) - Math.fround(orig - eps);
      w[i] = orig;
      const numeric = (up - down) / actual;
      const err = Math.abs(numeric - g[i]) / Math.max(1e-3, Math.abs(numeric) + Math.abs(g[i]));
      worst = Math.max(worst, err);
      assert.ok(err < 2e-3, `${spec.name}[${i}]: backprop ${g[i]} vs numeric ${numeric}`);
    }
  }
  assert.ok(worst < 2e-3);
});

test('the training forward pass gives the same loss as the decoding forward pass', async () => {
  const { be, model, tok } = gradModel();
  const ids = tok.encode('the quick brown fox');
  const decode = await model.nll(ids.slice(0, 1), ids.slice(1));
  const trainer = be.createTrainer(model.config, 32, []);
  const r = await trainer.loss([{ tokens: ids.slice(0, -1), targets: ids.slice(1) }]);
  assert.ok(Math.abs(r.sum / r.count - decode.loss) < 1e-5, `${r.sum / r.count} vs ${decode.loss}`);
});

test('AdamW changes only the trainable region and lowers the loss', async () => {
  const { be, model, tok } = gradModel();
  const seqs = [window(tok, 'hello hello hello')];
  const region = { rowStart: 2, rowEnd: 5, colStart: 1, colEnd: 4 };
  const trainer = be.createTrainer(model.config, 32, [{ name: 'layer_1.mlp.down', region }]);
  const before = Float32Array.from(be.data('layer_1.mlp.down'));
  const others = Float32Array.from(be.data('layer_0.mlp.down'));
  const first = await trainer.accumulate(seqs);
  for (let i = 0; i < 30; i++) {
    await trainer.accumulate(seqs);
    await trainer.update({ lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0, clip: 1 });
  }
  const last = await trainer.loss(seqs);
  assert.ok(last.sum / last.count < first.sum / first.count);
  const after = be.data('layer_1.mlp.down');
  const cols = be.shape('layer_1.mlp.down')[1];
  for (let i = 0; i < after.length; i++) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const inside = r >= 2 && r < 5 && c >= 1 && c < 4;
    if (!inside) assert.equal(after[i], before[i], `outside the region at ${r},${c}`);
  }
  assert.deepEqual([...be.data('layer_0.mlp.down')], [...others]);
});

test('microTrain learns a tiny dataset', async () => {
  const wsp = makeWsp({ dim: 32, heads: 2, kvHeads: 1, hidden: 32, layers: 1, std: 0.1 });
  await wsp.data.write('greet', ['hi there', 'hi there', 'hi there']);
  const r = await wsp.mutate.microTrain('all', 'greet', 120, { lr: 0.02, batchSize: 3, seed: 1 });
  assert.equal(r.method, 'AdamW with backpropagation; other tensors frozen');
  assert.ok(r.lossAfter < 1, `loss ${r.lossBefore} -> ${r.lossAfter}`);
  assert.ok(r.lossAfter < r.lossBefore / 4);
  const g = await wsp.eval.run('hi', ' there', { maxTokens: 6 });
  assert.equal(g.generated, ' there');
});

test('microTrain targets: layer prefixes, row and column blocks, and errors', async () => {
  const wsp = makeWsp();
  const resolved = wsp.resolveTargets([
    'layer_1',
    { tensor: 'layer_0.mlp.down', col_start: 20, col_end: 24 },
    { tensor: 'layer_0.mlp.gate', row_start: 20 },
  ]);
  assert.equal(resolved.filter((t) => t.name.startsWith('layer_1.')).length, 9);
  assert.deepEqual(resolved.find((t) => t.name === 'layer_0.mlp.down')!.region, { rowStart: 0, rowEnd: 16, colStart: 20, colEnd: 24 });
  assert.deepEqual(resolved.find((t) => t.name === 'layer_0.mlp.gate')!.region, { rowStart: 20, rowEnd: 24, colStart: 0, colEnd: 16 });
  assert.throws(() => wsp.resolveTargets([{ tensor: 'layer_0.mlp.down', col_start: 20, col_end: 99 }]), /outside/);
  assert.throws(() => wsp.resolveTargets(['layer_7']), /No tensor/);
  const before = await wsp.backend.read('layer_0.mlp.down');
  await wsp.data.write('d', ['abc abc abc']);
  await wsp.mutate.microTrain([{ tensor: 'layer_0.mlp.down', col_start: 20, col_end: 24 }], 'd', 5, { lr: 0.05 });
  const after = await wsp.backend.read('layer_0.mlp.down');
  for (let i = 0; i < after.length; i++) {
    if (i % 24 < 20) assert.equal(after[i], before[i]);
  }
});

test('datasets: write, append, sample, list', async () => {
  const wsp = makeWsp();
  await wsp.data.write('chat', ['a', 'b']);
  await wsp.data.write('chat', ['c'], true);
  assert.deepEqual(await wsp.data.get('chat'), ['a', 'b', 'c']);
  assert.deepEqual((await wsp.data.list()).map((d) => [d.name, d.texts]), [['chat', 3]]);
  assert.deepEqual((await wsp.data.sample('chat', 1, 5)).texts, ['b', 'c']);
  await assert.rejects(wsp.data.get('nope'), /No dataset/);
});

test('microTrain stops early when aborted and reports progress', async () => {
  const wsp = makeWsp();
  const controller = new AbortController();
  const seen: string[] = [];
  let steps = 0;
  const r = await wsp.mutate.microTrain('layer_1', ['abc abc abc', 'def def'], 1000, {
    lr: 0.01,
    signal: controller.signal,
    onProgress: (t) => {
      seen.push(t);
      if (++steps >= 1) controller.abort();
    },
  });
  assert.ok(r.stoppedEarly);
  assert.ok(r.steps < 1000);
  assert.ok(seen.length >= 1);
  assert.ok(Number.isFinite(r.lossAfter));
});
