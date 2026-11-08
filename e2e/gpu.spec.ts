/**
 * The WebGPU kernels against the CPU reference, in a real browser.
 * Each test runs inside the page, importing the library from Vite.
 */
import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  const hasGpu = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  test.skip(!hasGpu, 'no WebGPU adapter');
});

test('forward pass, NLL and generation match the CPU reference', async ({ page }) => {
  const r = await page.evaluate(async () => {
    const lib = (await import('/src/index.ts' as string)) as typeof import('../src/index.ts');
    // Grouped-query attention (4 query heads over 2 KV heads) and a separate classifier.
    const seed = lib.seedModel({ dim: 32, layers: 3, heads: 4, kvHeads: 2, hidden: 40, seqLen: 64, std: 0.3, seed: 11 });
    seed.config.sharedClassifier = false;
    seed.weights.set('lm_head', Float32Array.from({ length: 259 * 32 }, (_, i) => Math.sin(i) * 0.3));
    const tok = lib.Tokenizer.byteLevel();
    const make = (backend: any) => new lib.Model(backend, tok, seed.config, seed.weights, 'm');
    const gpu = make(await lib.WebGpuBackend.create());
    const cpu = make(new lib.CpuBackend());
    const tokens = tok.encode('Hello, WebGPU world!');
    const g = await gpu.forward().run(tokens, 0, { logits: true });
    const c = await cpu.forward().run(tokens, 0, { logits: true });
    let maxDiff = 0;
    let maxAbs = 0;
    for (let i = 0; i < c.logits!.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(c.logits![i] - g.logits![i]));
      maxAbs = Math.max(maxAbs, Math.abs(c.logits![i]));
    }
    const gn = await gpu.nll(tokens.slice(0, 3), tokens.slice(3));
    const cn = await cpu.nll(tokens.slice(0, 3), tokens.slice(3));
    const gg = await gpu.generate('Hi', { maxTokens: 12 });
    const cg = await cpu.generate('Hi', { maxTokens: 12 });
    return { maxDiff, maxAbs, gn, cn, gg: gg.tokens, cg: cg.tokens };
  });
  expect(r.maxAbs).toBeGreaterThan(0.1);
  expect(r.maxDiff).toBeLessThan(1e-3);
  expect(r.gn.tokens).toBe(r.cn.tokens);
  expect(Math.abs(r.gn.loss - r.cn.loss)).toBeLessThan(1e-3);
  expect(r.gg).toEqual(r.cg);
});

test('edit kernels match the CPU: stats, outliers, scatter, fill, copy, matvec, row norms', async ({ page }) => {
  const r = await page.evaluate(async () => {
    const lib = (await import('/src/index.ts' as string)) as typeof import('../src/index.ts');
    const gpu = await lib.WebGpuBackend.create();
    const cpu = new lib.CpuBackend();
    const shape: [number, number] = [300, 70];
    const data = Float32Array.from({ length: 300 * 70 }, (_, i) => Math.sin(i * 0.37) * (i % 13 === 0 ? 0 : 1));
    data[1234] = 40;
    data[777] = NaN;
    const out: Record<string, unknown> = {};
    const both = async (f: (b: any) => Promise<unknown>) => [await f(gpu), await f(cpu)];
    for (const b of [gpu, cpu]) b.alloc('t', shape, data);
    out.stats = await both((b) => b.stats('t'));
    out.outliers = await both(async (b) => {
      const h = await b.outliers('t', 0, 1, 5, 16);
      return { total: h.total, idx: [...h.indices].sort((x, y) => x - y) };
    });
    out.rowNorms = await both(async (b) => [...(await b.rowNorms('t'))].slice(0, 20));
    await Promise.all([gpu, cpu].map((b) => b.scatter('t', Uint32Array.of(5, 9000, 20999), Float32Array.of(1.5, -2, 3), 'add')));
    await Promise.all([gpu, cpu].map((b) => b.fill('t', { rowStart: 10, rowEnd: 20, colStart: 3, colEnd: 9 }, { kind: 'gaussian', mean: 0.5, std: 2, seed: 99 })));
    await Promise.all([gpu, cpu].map((b) => b.fill('t', { rowStart: 15, rowEnd: 40, colStart: 0, colEnd: 70 }, { kind: 'scale', factor: -0.5 })));
    for (const b of [gpu, cpu]) b.alloc('u', [50, 100]);
    await Promise.all([gpu, cpu].map((b) => b.copyRegion('t', 'u', { rowStart: 8, rowEnd: 30, colStart: 1, colEnd: 60 }, 3, 40)));
    out.matvec = await both(async (b) => [...(await b.matvec('t', Float32Array.from({ length: 70 }, (_, i) => i / 70)))].filter(Number.isFinite).slice(0, 50));
    const [gt, ct] = [await gpu.read('t'), await cpu.read('t')];
    const [gu, cu] = [await gpu.read('u'), await cpu.read('u')];
    let diff = 0;
    for (let i = 0; i < ct.length; i++) if (Number.isFinite(ct[i])) diff = Math.max(diff, Math.abs(ct[i] - gt[i]));
    for (let i = 0; i < cu.length; i++) diff = Math.max(diff, Math.abs(cu[i] - gu[i]));
    out.diff = diff;
    // A big tensor exercises the 2D dispatch grid (more than 65535 × 64 elements).
    const big = 70000 * 64 + 13;
    gpu.alloc('big', [1, big]);
    await gpu.fill('big', { rowStart: 0, rowEnd: 1, colStart: 0, colEnd: big }, { kind: 'constant', value: 2 });
    const s = await gpu.stats('big');
    out.big = { sum: s.sum, count: s.count, last: (await gpu.read('big', big - 1, 1))[0] };
    return out;
  });
  const [gs, cs] = r.stats as any[];
  expect(gs.count).toBe(cs.count);
  expect(gs.zeros).toBe(cs.zeros);
  expect(gs.nonFinite).toBe(1);
  expect(gs.max).toBeCloseTo(cs.max, 5);
  expect(Math.abs(gs.sum - cs.sum)).toBeLessThan(1e-2);
  expect((r.outliers as any[])[0]).toEqual((r.outliers as any[])[1]);
  const [gn, cn] = r.rowNorms as number[][];
  gn.forEach((v, i) => (Number.isNaN(cn[i]) ? expect(v).toBeNaN() : expect(v).toBeCloseTo(cn[i], 3)));
  const [gm, cm] = r.matvec as number[][];
  gm.forEach((v, i) => expect(v).toBeCloseTo(cm[i], 3));
  // f32 log/cos in the gaussian fill vs f64 on the CPU, on values up to ~±8.
  expect(r.diff).toBeLessThan(2e-3);
  expect(r.big).toEqual({ sum: 2 * (70000 * 64 + 13), count: 70000 * 64 + 13, last: 2 });
});

test('WSP on WebGPU: graft, prune, micro-train and OPFS checkpoints', async ({ page }) => {
  const r = await page.evaluate(async () => {
    const lib = (await import('/src/index.ts' as string)) as typeof import('../src/index.ts');
    const tok = lib.Tokenizer.byteLevel();
    const seed = lib.seedModel({ dim: 32, layers: 2, heads: 2, kvHeads: 1, hidden: 24, seqLen: 128, std: 0.25, seed: 5 });
    const store = new lib.OpfsStore(['wsp-kit-test', String(Math.random())]);
    const wsp = new lib.WSP(new lib.Model(await lib.WebGpuBackend.create(), tok, seed.config, seed.weights, 'seed'), { store });
    const logits = async () => [...(await wsp.model.forward().run(tok.encode('the cat'), 0, { logits: true })).logits!];
    const base = await logits();
    const ck = await wsp.state.checkpoint('base');
    await wsp.mutate.graftNode({ layer: 1, kind: 'mlp_neurons', count: 8, seed: 1 });
    await wsp.mutate.graftNode({ layer: 0, kind: 'attention_head', count: 1, kvHead: 0, seed: 2 });
    const afterGraft = await logits();
    const graftDiff = Math.max(...afterGraft.map((v, i) => Math.abs(v - base[i])));
    const prune = await wsp.mutate.pruneSemantic(['zzzz zzzz zzzz'], 1.5, { contrastDataset: ['the cat sat'], maxNeurons: 4 });
    const train = await wsp.mutate.microTrain('norm', ['aaaa aaaa'], 20, { lr: 0.01, seed: 3 });
    const shapeBefore = wsp.config.layers.map((l) => [l.nHeads, l.hidden]);
    await wsp.state.restore('base');
    const restored = await logits();
    const restoreDiff = Math.max(...restored.map((v, i) => Math.abs(v - base[i])));
    const list = await wsp.state.list();
    const anomalies = await wsp.inspect.findAnomalies('layer_0.mlp.gate', 3);
    return {
      store: store.kind,
      bytes: ck.bytes,
      graftDiff,
      pruned: prune.pruned.length,
      train: [train.lossBefore, train.lossAfter],
      shapeBefore,
      shapeAfter: wsp.config.layers.map((l) => [l.nHeads, l.hidden]),
      restoreDiff,
      list: list.map((c) => c.id),
      anomalies: anomalies.total,
    };
  });
  expect(r.store).toBe('opfs');
  expect(r.bytes).toBeGreaterThan(10000);
  expect(r.graftDiff).toBe(0);
  expect(r.pruned).toBeGreaterThan(0);
  expect(r.train[1]).toBeLessThanOrEqual(r.train[0]);
  expect(r.shapeBefore).toEqual([[3, 24], [2, 32]]);
  expect(r.shapeAfter).toEqual([[2, 24], [2, 24]]);
  expect(r.restoreDiff).toBe(0);
  expect(r.list).toEqual(['base']);
  expect(typeof r.anomalies).toBe('number');
});

test('backprop on WebGPU matches the CPU reference for every tensor', async ({ page }) => {
  const r = await page.evaluate(async () => {
    const lib = (await import('/src/index.ts' as string)) as typeof import('../src/index.ts');
    // GQA, a tied classifier, and a second layer with its own widths (as after grafts).
    const seed = lib.seedModel({ dim: 32, layers: 2, heads: 4, kvHeads: 2, hidden: 24, seqLen: 64, std: 0.3, seed: 8 });
    seed.config.layers[1] = { nHeads: 3, headKv: [1, 0, 1], hidden: 13 };
    for (const s of lib.tensorSpecs(seed.config)) {
      if (!s.name.startsWith('layer_1.')) continue;
      const n = s.shape[0] * s.shape[1];
      seed.weights.set(s.name, Float32Array.from({ length: n }, (_, i) => (s.name.endsWith('norm') ? 1 + 0.1 * Math.sin(i) : 0.3 * Math.sin(i * 1.3 + n))));
    }
    const tok = lib.Tokenizer.byteLevel();
    const gpu = new lib.Model(await lib.WebGpuBackend.create(), tok, seed.config, seed.weights, 'm');
    const cpu = new lib.Model(new lib.CpuBackend(), tok, seed.config, seed.weights, 'm');
    const win = (text: string) => {
      const ids = tok.encode(text, { eos: true });
      return { tokens: ids.slice(0, -1), targets: ids.slice(1) };
    };
    const seqs = [win('Hello, gradient!'), win('short'), win('WebGPU backprop test, a longer window here.')];
    const targets = gpu.specs().map((s) => ({ name: s.name, region: lib.fullRegion(s.shape) }));
    const gt = gpu.backend.createTrainer(gpu.config, 48, targets);
    const ct = cpu.backend.createTrainer(cpu.config, 48, targets);
    const gl = await gt.accumulate(seqs);
    const cl = await ct.accumulate(seqs);
    const gg = await gt.gradients();
    const cg = await ct.gradients();
    const worst: Record<string, number> = {};
    for (const [name, c] of cg) {
      const g = gg.get(name)!;
      let num = 0;
      let den = 0;
      for (let i = 0; i < c.length; i++) {
        num += (g[i] - c[i]) ** 2;
        den += c[i] ** 2;
      }
      worst[name] = Math.sqrt(num / Math.max(den, 1e-30));
    }
    // Scorer (batched) vs decode-path NLL on the GPU.
    const ids = tok.encode('The quick brown fox jumps');
    const decode = await gpu.forward().run(ids.slice(0, -1), 0, { targets: ids.slice(1) });
    const batched = await gpu.nll(ids.slice(0, 1), ids.slice(1));
    // One AdamW step on a block changes only that block.
    const region = { rowStart: 1, rowEnd: 4, colStart: 2, colEnd: 7 };
    const blockTrainer = gpu.backend.createTrainer(gpu.config, 48, [{ name: 'layer_1.mlp.down', region }]);
    const before = await gpu.backend.read('layer_1.mlp.down');
    await blockTrainer.accumulate(seqs);
    const u = await blockTrainer.update({ lr: 0.01, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, clip: 1 });
    const after = await gpu.backend.read('layer_1.mlp.down');
    let outside = 0;
    let inside = 0;
    for (let i = 0; i < after.length; i++) {
      const row = Math.floor(i / 13);
      const col = i % 13;
      const isIn = row >= 1 && row < 4 && col >= 2 && col < 7;
      if (after[i] !== before[i]) {
        if (isIn) inside++;
        else outside++;
      }
    }
    return { gl, cl, worst, decode: decode.nll!.sum / decode.nll!.count, batched: batched.loss, inside, outside, gradNorm: u.gradNorm };
  });
  expect(r.gl.count).toBe(r.cl.count);
  expect(Math.abs(r.gl.sum - r.cl.sum) / r.cl.sum).toBeLessThan(1e-4);
  for (const [name, err] of Object.entries(r.worst)) expect(err, name).toBeLessThan(1e-3);
  expect(Object.keys(r.worst)).toHaveLength(1 + 9 * 2 + 1);
  expect(Math.abs(r.batched - r.decode)).toBeLessThan(1e-4);
  expect(r.inside).toBe(15);
  expect(r.outside).toBe(0);
  expect(r.gradNorm).toBeGreaterThan(0);
});

test('microTrain on WebGPU learns a small dataset and matches the CPU', async ({ page }) => {
  const r = await page.evaluate(async () => {
    const lib = (await import('/src/index.ts' as string)) as typeof import('../src/index.ts');
    const run = async (backend: any) => {
      const tok = lib.Tokenizer.byteLevel();
      const seed = lib.seedModel({ dim: 32, layers: 1, heads: 2, kvHeads: 1, hidden: 32, seqLen: 64, std: 0.1, seed: 4 });
      const wsp = new lib.WSP(new lib.Model(backend, tok, seed.config, seed.weights, 'm'), {
        store: new lib.MemoryStore(),
        datasets: new lib.Datasets(new lib.MemoryStore()),
      });
      const r = await wsp.mutate.microTrain('all', ['hi there', 'hi there'], 60, { lr: 0.02, batchSize: 2, seed: 1 });
      const g = await wsp.eval.run('hi', ' there', { maxTokens: 6 });
      return { before: r.lossBefore, after: r.lossAfter, generated: g.generated };
    };
    return { gpu: await run(await lib.WebGpuBackend.create()), cpu: await run(new lib.CpuBackend()) };
  });
  expect(r.gpu.after).toBeLessThan(r.gpu.before / 3);
  expect(Math.abs(r.gpu.before - r.cpu.before)).toBeLessThan(1e-3);
  expect(Math.abs(r.gpu.after - r.cpu.after)).toBeLessThan(0.05);
  expect(r.gpu.generated).toBe(' there');
});
