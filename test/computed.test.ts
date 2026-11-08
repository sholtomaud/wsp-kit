import assert from 'node:assert/strict';
import { test } from 'node:test';
import { solveSpd } from '../src/computed.ts';
import { makeWsp } from './helpers.ts';

const prob = async (wsp: ReturnType<typeof makeWsp>, prompt: string, next: string) => {
  const r = await wsp.eval.run(prompt, next, { maxTokens: 1 });
  return Math.exp(-(r.loss as number));
};

test('Cholesky solve', () => {
  const A = Float64Array.of(4, 2, 0, 2, 5, 1, 0, 1, 3);
  const x = solveSpd(A, Float64Array.of(1, 2, 3), 3);
  const back = [0, 1, 2].map((i) => A[i * 3] * x[0] + A[i * 3 + 1] * x[1] + A[i * 3 + 2] * x[2]);
  back.forEach((v, i) => assert.ok(Math.abs(v - [1, 2, 3][i]) < 1e-9));
});

test('insertAssociation makes a prompt continue with a multi-token target, with no training', async () => {
  const wsp = makeWsp({ dim: 32, layers: 2, heads: 2, kvHeads: 1, hidden: 48, std: 0.2 });
  const before = await prob(wsp, 'capital of France:', ' Paris');
  // A tiny random classifier caps how sure it can be of any byte (~0.4–0.6 here), so aim for 0.35.
  const r = await wsp.mutate.insertAssociation({ prompt: 'capital of France:', target: ' Paris', layer: 1, goal: 0.35 });
  assert.equal(r.method, 'rank-one edit of mlp.down (ROME-style); no training loop');
  assert.ok(r.reachedGoal, JSON.stringify(r.tokens));
  assert.equal(r.tokens.length, 6); // byte-level: " Paris" is six tokens
  // prob() is the per-token geometric mean.
  assert.ok((await prob(wsp, 'capital of France:', ' Paris')) > before * 20);
  const g = await wsp.eval.run('capital of France:', undefined, { maxTokens: 6 });
  assert.equal(g.generated, ' Paris');
});

test('preserve texts shrink the edit\u2019s effect on their keys (ROME\u2019s C\u207b\u00b9k)', async () => {
  const preserve = ['the cat sat on the mat', 'a dog ran in the park', 'we like to sing songs'];
  const leak = async (withPreserve: boolean) => {
    const wsp = makeWsp({ dim: 32, layers: 2, heads: 2, kvHeads: 1, hidden: 48, std: 0.2, seed: 3 });
    const tok = wsp.model.tokenizer;
    const capturer = wsp.backend.createTrainer(wsp.config, 64, []);
    const keysOf = async (text: string) => {
      const ids = tok.encode(text);
      const cap = await capturer.capture({ tokens: ids, targets: ids.map(() => -1) }, 1);
      return Array.from({ length: ids.length }, (_, p) => cap.key.slice(p * 48, (p + 1) * 48));
    };
    const editKey = (await keysOf('zq')).at(-1)!;
    const keep = (await Promise.all(preserve.map(keysOf))).flat();
    const W0 = await wsp.backend.read('layer_1.mlp.down');
    await wsp.mutate.insertAssociation({ prompt: 'zq', target: 'x', layer: 1, goal: 0.5, ...(withPreserve ? { preserve } : {}) });
    const W1 = await wsp.backend.read('layer_1.mlp.down');
    const effect = (k: Float32Array) => {
      let s = 0;
      for (let i = 0; i < 32; i++) {
        let d = 0;
        for (let j = 0; j < 48; j++) d += (W1[i * 48 + j] - W0[i * 48 + j]) * k[j];
        s += d * d;
      }
      return Math.sqrt(s);
    };
    const onEdit = effect(editKey);
    return keep.reduce((m, k) => m + effect(k), 0) / keep.length / onEdit;
  };
  const plain = await leak(false);
  const guarded = await leak(true);
  assert.ok(guarded < plain / 2, `relative effect on preserved keys: ${guarded} with C vs ${plain} without`);
});

test('wireNeuron grafts a neuron that fires on the prompt and writes the token', async () => {
  const wsp = makeWsp({ dim: 32, layers: 2, heads: 2, kvHeads: 1, hidden: 16, std: 0.2 });
  const r = await wsp.mutate.wireNeuron({ layer: 1, prompt: 'Hello', token: '!', preserve: ['xyz xyz'], goal: 0.4 });
  assert.equal(r.grafted, true);
  assert.equal(r.neuron, 16);
  assert.equal(wsp.config.layers[1].hidden, 17);
  assert.ok(r.reachedGoal, JSON.stringify(r));
  assert.ok(Math.abs((r.activationOnPrompt as number) - 4 / (1 + Math.exp(-4)) * 4) < 0.5, `activation ${r.activationOnPrompt}`);
  assert.ok(r.probabilityAfter > r.probabilityBefore * 10);
  const n = await wsp.inspect.neuron(1, 16, 3);
  assert.ok(n.outputNorm > 0);
});

test('computed edits explain what is missing', async () => {
  const wsp = makeWsp({ hidden: 0 });
  await assert.rejects(wsp.mutate.insertAssociation({ prompt: 'a', target: 'b', layer: 0 }), /no MLP neurons/);
  await assert.rejects(wsp.mutate.insertAssociation({ prompt: 'a', target: '', layer: 1 }), /no tokens|no MLP/);
});
