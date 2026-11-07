import assert from 'node:assert/strict';
import { test } from 'node:test';
import { makeWsp, maxAbsDiff } from './helpers.ts';

const logitsOf = async (wsp: ReturnType<typeof makeWsp>, text: string) => {
  const tokens = wsp.model.tokenizer.encode(text);
  return (await wsp.model.forward().run(tokens, 0, { logits: true })).logits!;
};

test('the forward pass is deterministic and depends on the prompt', async () => {
  const wsp = makeWsp();
  const a = await logitsOf(wsp, 'hello');
  const b = await logitsOf(wsp, 'hello');
  const c = await logitsOf(wsp, 'help!');
  assert.equal(maxAbsDiff(a, b), 0);
  assert.ok(maxAbsDiff(a, c) > 1e-4);
});

test('summary reports shape, sparsity and moments', async () => {
  const wsp = makeWsp();
  const s = await wsp.inspect.summary('layer_0.mlp.gate');
  assert.ok('shape' in s);
  assert.deepEqual(s.shape, [24, 16]);
  assert.equal(s.sparsity, 0);
  assert.ok(Math.abs(s.std - 0.2) < 0.03);
  const layer = await wsp.inspect.summary('layer_1');
  assert.ok('tensors' in layer && layer.tensors.length === 9);
  await assert.rejects(wsp.inspect.summary('layer_1.mlp.gat'), /Did you mean/);
});

test('slice renders a bounded Markdown grid', async () => {
  const wsp = makeWsp();
  await wsp.mutate.patchSparse('layer_0.attn.wq', [[1, 2, 0.5]]);
  const s = await wsp.inspect.slice('layer_0.attn.wq', { rowStart: 0, rowEnd: 3, colStart: 0, colEnd: 4 });
  const lines = s.markdown.split('\n');
  assert.equal(lines.length, 5);
  assert.match(lines[3], /^\| 1 \| .* \| 0\.5000 \|/);
  const capped = await wsp.inspect.slice('tok_embeddings', { rowStart: 0, rowEnd: 999, colStart: 0, colEnd: 999 });
  assert.deepEqual(capped.bounds, { rowStart: 0, rowEnd: 32, colStart: 0, colEnd: 16 });
});

test('patchSparse writes exactly the given coordinates', async () => {
  const wsp = makeWsp();
  const before = await wsp.backend.read('layer_1.mlp.down');
  const r = await wsp.mutate.patchSparse('layer_1.mlp.down', [[0, 0, 9], { row: 3, col: 5, value: -2 }, [0, 0, 7]]);
  assert.equal(r.applied, 2, 'duplicate coordinates are merged (last write wins)');
  const after = await wsp.backend.read('layer_1.mlp.down');
  const cols = 24;
  assert.equal(after[0], 7);
  assert.equal(after[3 * cols + 5], -2);
  let changed = 0;
  for (let i = 0; i < after.length; i++) if (after[i] !== before[i]) changed++;
  assert.equal(changed, 2);
  await wsp.mutate.patchSparse('layer_1.mlp.down', [[0, 0, 1]], 'add');
  assert.equal((await wsp.backend.read('layer_1.mlp.down'))[0], 8);
  await assert.rejects(wsp.mutate.patchSparse('layer_1.mlp.down', [[16, 0, 1]]), /outside/);
});

test('findAnomalies finds planted outliers and dead rows', async () => {
  const wsp = makeWsp();
  await wsp.mutate.patchSparse('layer_0.mlp.up', [[4, 7, 50], [9, 1, -40]]);
  const v = await wsp.inspect.findAnomalies('layer_0.mlp.up', 6);
  assert.equal(v.total, 2);
  assert.deepEqual(v.hits!.map((h) => [h.row, h.col]), [[4, 7], [9, 1]]);
  await wsp.backend.fill('layer_0.mlp.up', { rowStart: 11, rowEnd: 12, colStart: 0, colEnd: 16 }, { kind: 'constant', value: 0 });
  const rows = await wsp.inspect.findAnomalies('layer_0.mlp.up', 2.5, { mode: 'rows' });
  assert.ok(rows.rows!.some((r) => r.row === 11 && r.norm === 0), 'dead row reported');
  assert.ok(rows.rows!.some((r) => r.row === 4));
});

test('grafting MLP neurons with zero output weights preserves the function', async () => {
  const wsp = makeWsp();
  const before = await logitsOf(wsp, 'the cat');
  const g = await wsp.mutate.graftNode({ layer: 1, kind: 'mlp_neurons', count: 5, seed: 1 });
  assert.deepEqual(g.newIndices, [24, 28]);
  assert.deepEqual(wsp.backend.shape('layer_1.mlp.gate'), [29, 16]);
  assert.deepEqual(wsp.backend.shape('layer_1.mlp.down'), [16, 29]);
  assert.equal(maxAbsDiff(before, await logitsOf(wsp, 'the cat')), 0);
  // The new gate rows are random, not zero.
  const s = await wsp.inspect.summary('layer_1.mlp.gate');
  assert.ok('sparsity' in s && s.sparsity === 0);
});

test('cloning a neuron with split outputs preserves the function (Net2Net)', async () => {
  const wsp = makeWsp();
  const before = await logitsOf(wsp, 'hello');
  await wsp.mutate.graftNode({ layer: 0, kind: 'mlp_neurons', count: 2, init: 'clone', source: 3, noise: 0, seed: 2 });
  assert.ok(maxAbsDiff(before, await logitsOf(wsp, 'hello')) < 1e-5);
});

test('grafting attention heads: zero output preserves, gaussian output changes', async () => {
  const wsp = makeWsp();
  const before = await logitsOf(wsp, 'abc');
  const g = await wsp.mutate.graftNode({ layer: 0, kind: 'attention_head', count: 1, kvHead: 0, seed: 3 });
  assert.equal(g.layerNow.heads, 3);
  assert.deepEqual(wsp.config.layers[0].headKv, [0, 0, 0]);
  assert.deepEqual(wsp.backend.shape('layer_0.attn.wq'), [24, 16]);
  assert.deepEqual(wsp.backend.shape('layer_0.attn.wo'), [16, 24]);
  assert.equal(maxAbsDiff(before, await logitsOf(wsp, 'abc')), 0);
  await wsp.mutate.graftNode({ layer: 0, kind: 'attention_head', output: 'gaussian', seed: 4 });
  assert.ok(maxAbsDiff(before, await logitsOf(wsp, 'abc')) > 1e-4);
  await assert.rejects(wsp.mutate.graftNode({ layer: 0, kind: 'attention_head', kvHead: 5 }), /kvHead/);
  assert.equal(wsp.config.layers[0].nHeads, 4, 'a rejected graft changes nothing');
});

test('a model can grow from zero capacity', async () => {
  const wsp = makeWsp({ heads: 0, hidden: 0, kvHeads: 1, layers: 1 });
  const before = await logitsOf(wsp, 'hi');
  await wsp.mutate.graftNode({ layer: 0, kind: 'attention_head', count: 2, output: 'gaussian', seed: 5 });
  await wsp.mutate.graftNode({ layer: 0, kind: 'mlp_neurons', count: 8, output: 'gaussian', seed: 6 });
  assert.deepEqual(wsp.config.layers[0], { nHeads: 2, headKv: [0, 0], hidden: 8 });
  assert.ok(maxAbsDiff(before, await logitsOf(wsp, 'hi')) > 1e-5);
});

test('pruneSemantic zeroes the neurons specific to the trigger texts', async () => {
  const wsp = makeWsp();
  const dry = await wsp.mutate.pruneSemantic(['zzzz zzzz'], 1.5, { contrastDataset: ['hello there'], maxNeurons: 3, dryRun: true });
  assert.equal(dry.pruned.length, 0);
  assert.ok(dry.neurons.length > 0 && dry.neurons.length <= 3);
  const real = await wsp.mutate.pruneSemantic(['zzzz zzzz'], 1.5, { contrastDataset: ['hello there'], maxNeurons: 3 });
  assert.deepEqual(real.pruned, dry.neurons.map(({ layer, neuron }) => ({ layer, neuron })));
  for (const { layer, neuron } of real.pruned) {
    const gate = await wsp.backend.read(`layer_${layer}.mlp.gate`, neuron * 16, 16);
    assert.ok(gate.every((v) => v === 0));
  }
  const again = await wsp.mutate.pruneSemantic(['zzzz zzzz'], 1.5, { contrastDataset: ['hello there'], maxNeurons: 3, dryRun: true });
  const prunedKeys = new Set(real.pruned.map((p) => `${p.layer}:${p.neuron}`));
  assert.ok(again.neurons.every((n) => !prunedKeys.has(`${n.layer}:${n.neuron}`)), 'dead neurons are not chosen again');
});

test('checkpoint and restore round-trip values and shape', async () => {
  const wsp = makeWsp();
  const before = await logitsOf(wsp, 'cat');
  await wsp.state.checkpoint('base');
  await wsp.mutate.graftNode({ layer: 0, kind: 'mlp_neurons', count: 3, output: 'gaussian', seed: 9 });
  await wsp.mutate.patchSparse('tok_embeddings', [[1, 1, 100]]);
  assert.ok(maxAbsDiff(before, await logitsOf(wsp, 'cat')) > 1e-3);
  await wsp.state.restore('base');
  assert.equal(wsp.config.layers[0].hidden, 24);
  assert.equal(maxAbsDiff(before, await logitsOf(wsp, 'cat')), 0);
  assert.deepEqual((await wsp.state.list()).map((c) => c.id), ['base']);
  await assert.rejects(wsp.state.checkpoint('../evil'), /must be/);
});

test('eval.run reports loss, perplexity and the change since last time', async () => {
  const wsp = makeWsp();
  const a = await wsp.eval.run('The cat', ' sat', { maxTokens: 4 });
  assert.equal(typeof a.generated, 'string');
  assert.ok((a.loss as number) > 0);
  assert.ok(Math.abs((a.perplexity as number) - Math.exp(a.loss as number)) / (a.perplexity as number) < 1e-3);
  assert.equal(a.scoredTokens, 4);
  assert.equal(a.previousLoss, undefined);
  const b = await wsp.eval.run('The cat', ' sat', { maxTokens: 4 });
  assert.equal(b.delta, 0);
});

test('microTrain lowers the loss without touching other tensors', async () => {
  const wsp = makeWsp();
  const other = await wsp.backend.read('layer_0.mlp.gate');
  const r = await wsp.mutate.microTrain('norm', ['aaaa aaaa aaaa'], 40, { sigma: 0.5, seed: 1 });
  assert.ok(r.lossAfter <= r.lossBefore);
  assert.ok(r.accepted > 0, 'some steps improved the loss');
  assert.equal(maxAbsDiff(other, await wsp.backend.read('layer_0.mlp.gate')), 0);
  assert.equal(wsp.backend.has('norm#backup'), false);
});

test('neuron inspection names tokens', async () => {
  const wsp = makeWsp();
  const n = await wsp.inspect.neuron(0, 2, 3);
  assert.equal(n.promotes.length, 3);
  assert.equal(n.activatedBy.length, 3);
  assert.equal(typeof n.promotes[0].piece, 'string');
  assert.equal(wsp.backend.has('__neuron_col'), false);
});
