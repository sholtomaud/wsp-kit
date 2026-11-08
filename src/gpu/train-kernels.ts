/**
 * WGSL for training: a batched forward pass over a window of T tokens that
 * keeps its activations, the backward pass, and AdamW.
 *
 * Layouts: activations are row-major [T, width]; attention weights are
 * [head, t, s] with stride T. Elementwise kernels use the 2D grid
 * idx = gid.y·strideX + gid.x. (WGSL reserves many words, `target` among
 * them, so targets are called `goal` here.)
 */

export const TRAIN_KERNELS = {
  /** C[M,N] (+)= A[M,K] · B[K,N], with A or B optionally stored transposed. 16×16 tiles. */
  tMatmul: /* wgsl */ `
struct P { M: u32, N: u32, K: u32, transA: u32, transB: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wid.y * 16u + lid.y;
  let col = wid.x * 16u + lid.x;
  var acc = 0.0;
  let tiles = (p.K + 15u) / 16u;
  for (var t = 0u; t < tiles; t++) {
    let ka = t * 16u + lid.x;
    var av = 0.0;
    if (row < p.M && ka < p.K) {
      if (p.transA == 1u) { av = a[ka * p.M + row]; } else { av = a[row * p.K + ka]; }
    }
    tileA[lid.y][lid.x] = av;
    let kb = t * 16u + lid.y;
    var bv = 0.0;
    if (kb < p.K && col < p.N) {
      if (p.transB == 1u) { bv = b[col * p.K + kb]; } else { bv = b[kb * p.N + col]; }
    }
    tileB[lid.y][lid.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { acc += tileA[lid.y][k] * tileB[k][lid.x]; }
    workgroupBarrier();
  }
  if (row < p.M && col < p.N) {
    let i = row * p.N + col;
    if (p.accumulate == 1u) { c[i] = c[i] + acc; } else { c[i] = acc; }
  }
}`,

  tGather: /* wgsl */ `
struct P { T: u32, dim: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> emb: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.y * p.strideX + g.x;
  if (idx >= p.T * p.dim) { return; }
  let t = idx / p.dim;
  x[idx] = emb[tokens[t] * p.dim + idx % p.dim];
}`,

  /** One workgroup per row: y = w ⊙ x · rstd, keeping rstd. */
  tRmsF: /* wgsl */ `
struct P { n: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> rstd: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) i: u32) {
  let o = wid.x * p.n;
  var acc = 0.0;
  for (var j = i; j < p.n; j += 64u) { acc += x[o + j] * x[o + j]; }
  red[i] = acc;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (i < k) { red[i] += red[i + k]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(p.n) + p.eps);
  if (i == 0u) { rstd[wid.x] = r; }
  for (var j = i; j < p.n; j += 64u) { y[o + j] = w[j] * (r * x[o + j]); }
}`,

  /** One workgroup per row: dx (+)= r·w⊙dy − x·r³·(Σ w·dy·x)/n. */
  tRmsB: /* wgsl */ `
struct P { n: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) i: u32) {
  let o = wid.x * p.n;
  var acc = 0.0;
  for (var j = i; j < p.n; j += 64u) { acc += w[j] * dy[o + j] * x[o + j]; }
  red[i] = acc;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (i < k) { red[i] += red[i + k]; }
    workgroupBarrier();
  }
  let r = rstd[wid.x];
  let s = red[0];
  for (var j = i; j < p.n; j += 64u) {
    let v = r * w[j] * dy[o + j] - x[o + j] * r * r * r * s / f32(p.n);
    if (p.accumulate == 1u) { dx[o + j] = dx[o + j] + v; } else { dx[o + j] = v; }
  }
}`,

  /** dw[j] += Σ_t dy[t, j]·x[t, j]·rstd[t] */
  tRmsBW: /* wgsl */ `
struct P { T: u32, n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> rstd: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  if (j >= p.n) { return; }
  var acc = 0.0;
  for (var t = 0u; t < p.T; t++) { acc += dy[t * p.n + j] * x[t * p.n + j] * rstd[t]; }
  dw[j] = dw[j] + acc;
}`,

  /** Rotates adjacent pairs of a [T, width] buffer by ±position·freq. */
  tRope: /* wgsl */ `
struct P { T: u32, width: u32, headSize: u32, strideX: u32, theta: f32, sign: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.y * p.strideX + g.x;
  let halfW = p.width / 2u;
  if (idx >= p.T * halfW) { return; }
  let t = idx / halfW;
  let i = (idx % halfW) * 2u;
  let val = f32(t) / pow(p.theta, f32(i % p.headSize) / f32(p.headSize));
  let c = cos(val);
  let s = p.sign * sin(val);
  let o = t * p.width + i;
  let v0 = buf[o];
  let v1 = buf[o + 1u];
  buf[o] = v0 * c - v1 * s;
  buf[o + 1u] = v0 * s + v1 * c;
}`,

  /** Workgroup (t, h): causal softmax weights over s ≤ t, kept in att, and the head output. */
  tAttnF: /* wgsl */ `
struct P { T: u32, headSize: u32, qd: u32, kv: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> headKv: array<u32>;
@group(0) @binding(5) var<storage, read_write> att: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) i: u32) {
  let t = wid.x;
  let h = wid.y;
  let hs = p.headSize;
  let n = t + 1u;
  let qo = t * p.qd + h * hs;
  let ko = headKv[h] * hs;
  let ao = (h * p.T + t) * p.T;
  let scale = inverseSqrt(f32(hs));
  var mx = -3.4e38;
  for (var s = i; s < n; s += 64u) {
    var d = 0.0;
    for (var e = 0u; e < hs; e++) { d += q[qo + e] * k[s * p.kv + ko + e]; }
    d *= scale;
    att[ao + s] = d;
    mx = max(mx, d);
  }
  red[i] = mx;
  workgroupBarrier();
  for (var r = 32u; r > 0u; r >>= 1u) {
    if (i < r) { red[i] = max(red[i], red[i + r]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var s = i; s < n; s += 64u) {
    let e = exp(att[ao + s] - m);
    att[ao + s] = e;
    sum += e;
  }
  red[i] = sum;
  workgroupBarrier();
  for (var r = 32u; r > 0u; r >>= 1u) {
    if (i < r) { red[i] += red[i + r]; }
    workgroupBarrier();
  }
  let total = red[0];
  for (var s = i; s < n; s += 64u) { att[ao + s] = att[ao + s] / total; }
  storageBarrier();
  workgroupBarrier();
  for (var e = i; e < hs; e += 64u) {
    var acc = 0.0;
    for (var s = 0u; s < n; s++) { acc += att[ao + s] * v[s * p.kv + ko + e]; }
    o[qo + e] = acc;
  }
}`,

  /** dV[s, c] = Σ over query heads on c's KV head, Σ_{t ≥ s} att[h, t, s]·dO[t, h, d] */
  tAttnDV: /* wgsl */ `
struct P { T: u32, headSize: u32, qd: u32, kv: u32, nHeads: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> att: array<f32>;
@group(0) @binding(2) var<storage, read> dO: array<f32>;
@group(0) @binding(3) var<storage, read> headKv: array<u32>;
@group(0) @binding(4) var<storage, read_write> dV: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.y * p.strideX + g.x;
  if (idx >= p.T * p.kv) { return; }
  let s = idx / p.kv;
  let c = idx % p.kv;
  let kvh = c / p.headSize;
  let d = c % p.headSize;
  var acc = 0.0;
  for (var h = 0u; h < p.nHeads; h++) {
    if (headKv[h] != kvh) { continue; }
    for (var t = s; t < p.T; t++) { acc += att[(h * p.T + t) * p.T + s] * dO[t * p.qd + h * p.headSize + d]; }
  }
  dV[idx] = acc;
}`,

  /** Workgroup (t, h): dS[h, t, s] = att·(dP − Σ att·dP), with dP[s] = dO[t, h]·v[s]. */
  tAttnDS: /* wgsl */ `
struct P { T: u32, headSize: u32, qd: u32, kv: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> att: array<f32>;
@group(0) @binding(2) var<storage, read> dO: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> headKv: array<u32>;
@group(0) @binding(5) var<storage, read_write> dS: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) i: u32) {
  let t = wid.x;
  let h = wid.y;
  let hs = p.headSize;
  let n = t + 1u;
  let go = t * p.qd + h * hs;
  let ko = headKv[h] * hs;
  let ao = (h * p.T + t) * p.T;
  var part = 0.0;
  for (var s = i; s < n; s += 64u) {
    var d = 0.0;
    for (var e = 0u; e < hs; e++) { d += dO[go + e] * v[s * p.kv + ko + e]; }
    dS[ao + s] = d;
    part += att[ao + s] * d;
  }
  red[i] = part;
  workgroupBarrier();
  for (var r = 32u; r > 0u; r >>= 1u) {
    if (i < r) { red[i] += red[i + r]; }
    workgroupBarrier();
  }
  let dsum = red[0];
  for (var s = i; s < n; s += 64u) { dS[ao + s] = att[ao + s] * (dS[ao + s] - dsum); }
}`,

  /** dQ[t, c] = scale · Σ_{s ≤ t} dS[h, t, s]·k[s] */
  tAttnDQ: /* wgsl */ `
struct P { T: u32, headSize: u32, qd: u32, kv: u32, strideX: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dS: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> headKv: array<u32>;
@group(0) @binding(4) var<storage, read_write> dQ: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.y * p.strideX + g.x;
  if (idx >= p.T * p.qd) { return; }
  let t = idx / p.qd;
  let c = idx % p.qd;
  let h = c / p.headSize;
  let d = c % p.headSize;
  let ko = headKv[h] * p.headSize + d;
  var acc = 0.0;
  for (var s = 0u; s <= t; s++) { acc += dS[(h * p.T + t) * p.T + s] * k[s * p.kv + ko]; }
  dQ[idx] = acc * p.scale;
}`,

  /** dK[s, c] = scale · Σ over query heads on c's KV head, Σ_{t ≥ s} dS[h, t, s]·q[t, h] */
  tAttnDK: /* wgsl */ `
struct P { T: u32, headSize: u32, qd: u32, kv: u32, nHeads: u32, strideX: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dS: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> headKv: array<u32>;
@group(0) @binding(4) var<storage, read_write> dK: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.y * p.strideX + g.x;
  if (idx >= p.T * p.kv) { return; }
  let s = idx / p.kv;
  let c = idx % p.kv;
  let kvh = c / p.headSize;
  let d = c % p.headSize;
  var acc = 0.0;
  for (var h = 0u; h < p.nHeads; h++) {
    if (headKv[h] != kvh) { continue; }
    for (var t = s; t < p.T; t++) { acc += dS[(h * p.T + t) * p.T + s] * q[t * p.qd + h * p.headSize + d]; }
  }
  dK[idx] = acc * p.scale;
}`,

  tSwigluF: /* wgsl */ `
struct P { n: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> h: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.y * p.strideX + g.x;
  if (i >= p.n) { return; }
  let a = gate[i];
  h[i] = a / (1.0 + exp(-a)) * up[i];
}`,

  tSwigluB: /* wgsl */ `
struct P { n: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dh: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read_write> dg: array<f32>;
@group(0) @binding(5) var<storage, read_write> du: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.y * p.strideX + g.x;
  if (i >= p.n) { return; }
  let a = gate[i];
  let sg = 1.0 / (1.0 + exp(-a));
  dg[i] = dh[i] * up[i] * sg * (1.0 + a * (1.0 - sg));
  du[i] = dh[i] * a * sg;
}`,

  /** One workgroup per row: adds the row's NLL to lossAcc and overwrites logits with scale·(softmax − onehot). */
  tCrossEntropy: /* wgsl */ `
struct P { vocab: u32 }
struct Seq { scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> seq: Seq;
@group(0) @binding(2) var<storage, read> goals: array<i32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@group(0) @binding(4) var<storage, read_write> lossAcc: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) i: u32) {
  let row = wid.x;
  let o = row * p.vocab;
  let goal = goals[row];
  let valid = goal >= 0;
  let gi = u32(max(goal, 0));
  var mx = -3.4e38;
  for (var j = i; j < p.vocab; j += 256u) { mx = max(mx, logits[o + j]); }
  red[i] = mx;
  workgroupBarrier();
  for (var r = 128u; r > 0u; r >>= 1u) {
    if (i < r) { red[i] = max(red[i], red[i + r]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = i; j < p.vocab; j += 256u) { sum += exp(logits[o + j] - m); }
  red[i] = sum;
  workgroupBarrier();
  for (var r = 128u; r > 0u; r >>= 1u) {
    if (i < r) { red[i] += red[i + r]; }
    workgroupBarrier();
  }
  let lse = m + log(red[0]);
  let picked = logits[o + gi];
  storageBarrier();
  workgroupBarrier();
  if (i == 0u && valid) { lossAcc[row] = lossAcc[row] + lse - picked; }
  for (var j = i; j < p.vocab; j += 256u) {
    var gr = 0.0;
    if (valid) {
      gr = exp(logits[o + j] - lse);
      if (j == gi) { gr -= 1.0; }
      gr *= seq.scale;
    }
    logits[o + j] = gr;
  }
}`,

  /** dE[token[t], j] += dX[t, j]; one thread per column, so repeated tokens never race. */
  tEmbB: /* wgsl */ `
struct P { T: u32, dim: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> dx: array<f32>;
@group(0) @binding(3) var<storage, read_write> de: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  if (j >= p.dim) { return; }
  for (var t = 0u; t < p.T; t++) {
    let i = tokens[t] * p.dim + j;
    de[i] = de[i] + dx[t * p.dim + j];
  }
}`,

  tCopy: /* wgsl */ `
struct P { n: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.y * p.strideX + g.x;
  if (i >= p.n) { return; }
  y[i] = x[i];
}`,

  /** AdamW on one block of a tensor (decoupled weight decay, bias-corrected moments). */
  tAdamW: /* wgsl */ `
struct P { cols: u32, r0: u32, c0: u32, nCols: u32, n: u32, strideX: u32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, bc1: f32, bc2: f32, gscale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.y * p.strideX + g.x;
  if (idx >= p.n) { return; }
  let i = (p.r0 + idx / p.nCols) * p.cols + p.c0 + idx % p.nCols;
  let gi = grad[i] * p.gscale;
  let mi = p.b1 * m[i] + (1.0 - p.b1) * gi;
  let vi = p.b2 * v[i] + (1.0 - p.b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  w[i] = w[i] - p.lr * ((mi / p.bc1) / (sqrt(vi / p.bc2) + p.eps) + p.wd * w[i]);
}`,
} as const;

export type TrainKernelName = keyof typeof TRAIN_KERNELS;
