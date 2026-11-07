/**
 * The WGSL kernels. Each one declares its own bindings (all of them used, as
 * `layout: 'auto'` requires) and a Params uniform of 4-byte scalars.
 *
 * Elementwise kernels run 64 threads per workgroup over a 2D grid, so a
 * dispatch can cover more than 65535 × 64 elements: idx = gid.y·strideX + gid.x.
 */
import { PCG_WGSL } from '../model/random.ts';

const STEP = /* wgsl */ `struct Step { token: u32, pos: u32, goal: i32, track: u32 }`;

export const KERNELS = {
  fill: /* wgsl */ `
${PCG_WGSL}
struct P { cols: u32, r0: u32, c0: u32, nCols: u32, n: u32, mode: u32, seed: u32, strideX: u32, a: f32, b: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> t: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.y * p.strideX + g.x;
  if (idx >= p.n) { return; }
  let i = (p.r0 + idx / p.nCols) * p.cols + p.c0 + idx % p.nCols;
  switch p.mode {
    case 0u: { t[i] = p.a; }
    case 1u: { t[i] = p.a + p.b * gaussian(p.seed, i); }
    case 2u: { t[i] = t[i] + p.b * gaussian(p.seed, i); }
    default: { t[i] = t[i] * p.a; }
  }
}`,

  copyRegion: /* wgsl */ `
struct P { srcCols: u32, dstCols: u32, sr0: u32, sc0: u32, dr0: u32, dc0: u32, nCols: u32, n: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.y * p.strideX + g.x;
  if (idx >= p.n) { return; }
  let r = idx / p.nCols;
  let c = idx % p.nCols;
  dst[(p.dr0 + r) * p.dstCols + p.dc0 + c] = src[(p.sr0 + r) * p.srcCols + p.sc0 + c];
}`,

  scatter: /* wgsl */ `
struct P { n: u32, mode: u32, len: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> t: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.y * p.strideX + g.x;
  if (k >= p.n) { return; }
  let i = idx[k];
  if (i >= p.len) { return; }
  if (p.mode == 0u) { t[i] = vals[k]; } else { t[i] = t[i] + vals[k]; }
}`,

  /** Partial sums per workgroup: sum, sumSq, min, max, zeros, nonFinite (8 floats each). */
  stats: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> t: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> sSum: array<f32, 256>;
var<workgroup> sSq: array<f32, 256>;
var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;
var<workgroup> sZero: array<u32, 256>;
var<workgroup> sBad: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  var sum = 0.0; var sq = 0.0; var mn = 3.4e38; var mx = -3.4e38; var zeros = 0u; var bad = 0u;
  let stride = nwg.x * 256u;
  for (var i = wid.x * 256u + li; i < p.n; i += stride) {
    let x = t[i];
    if ((bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u) { bad += 1u; continue; }
    sum += x; sq += x * x; mn = min(mn, x); mx = max(mx, x);
    if (x == 0.0) { zeros += 1u; }
  }
  sSum[li] = sum; sSq[li] = sq; sMin[li] = mn; sMax[li] = mx; sZero[li] = zeros; sBad[li] = bad;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) {
      sSum[li] += sSum[li + k]; sSq[li] += sSq[li + k];
      sMin[li] = min(sMin[li], sMin[li + k]); sMax[li] = max(sMax[li], sMax[li + k]);
      sZero[li] += sZero[li + k]; sBad[li] += sBad[li + k];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let o = wid.x * 8u;
    out[o] = sSum[0]; out[o + 1u] = sSq[0]; out[o + 2u] = sMin[0]; out[o + 3u] = sMax[0];
    out[o + 4u] = f32(sZero[0]); out[o + 5u] = f32(sBad[0]);
  }
}`,

  rowNorms: /* wgsl */ `
struct P { rows: u32, cols: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> t: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let r = g.y * p.strideX + g.x;
  if (r >= p.rows) { return; }
  var s = 0.0;
  for (var j = 0u; j < p.cols; j++) { let x = t[r * p.cols + j]; s += x * x; }
  out[r] = sqrt(s);
}`,

  outliers: /* wgsl */ `
struct P { n: u32, cap: u32, strideX: u32, mean: f32, invStd: f32, z: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> t: array<f32>;
@group(0) @binding(2) var<storage, read_write> oi: array<u32>;
@group(0) @binding(3) var<storage, read_write> ov: array<f32>;
@group(0) @binding(4) var<storage, read_write> count: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.y * p.strideX + g.x;
  if (i >= p.n) { return; }
  let x = t[i];
  let bad = (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u;
  if (bad || abs(x - p.mean) * p.invStd >= p.z) {
    let k = atomicAdd(&count[0], 1u);
    if (k < p.cap) { oi[k] = i; ov[k] = x; }
  }
}`,

  // ---- forward pass ----

  embed: /* wgsl */ `
${STEP}
struct P { dim: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read> emb: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.dim) { return; }
  x[g.x] = emb[step.token * p.dim + g.x];
}`,

  rmsnorm: /* wgsl */ `
struct P { n: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
var<workgroup> s: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i: u32) {
  var acc = 0.0;
  for (var j = i; j < p.n; j += 256u) { acc += x[j] * x[j]; }
  s[i] = acc;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (i < k) { s[i] += s[i + k]; }
    workgroupBarrier();
  }
  let scale = inverseSqrt(s[0] / f32(p.n) + p.eps);
  for (var j = i; j < p.n; j += 256u) { o[j] = w[j] * (scale * x[j]); }
}`,

  /** y[row + pos·posStride] (+)= W[row, :] · x */
  matvec: /* wgsl */ `
${STEP}
struct P { rows: u32, cols: u32, accumulate: u32, posStride: u32, strideX: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let row = g.y * p.strideX + g.x;
  if (row >= p.rows) { return; }
  var s = 0.0;
  let base = row * p.cols;
  for (var j = 0u; j < p.cols; j++) { s += w[base + j] * x[j]; }
  let o = row + step.pos * p.posStride;
  if (p.accumulate == 1u) { y[o] = y[o] + s; } else { y[o] = s; }
}`,

  /** Rotary embeddings on q and on this position's slot of the key cache. */
  rope: /* wgsl */ `
${STEP}
struct P { qDim: u32, kvDim: u32, headSize: u32, theta: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> kc: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x * 2u;
  if (i >= max(p.qDim, p.kvDim)) { return; }
  let freq = 1.0 / pow(p.theta, f32(i % p.headSize) / f32(p.headSize));
  let val = f32(step.pos) * freq;
  let c = cos(val);
  let s = sin(val);
  if (i < p.qDim) {
    let v0 = q[i]; let v1 = q[i + 1u];
    q[i] = v0 * c - v1 * s; q[i + 1u] = v0 * s + v1 * c;
  }
  if (i < p.kvDim) {
    let o = step.pos * p.kvDim + i;
    let v0 = kc[o]; let v1 = kc[o + 1u];
    kc[o] = v0 * c - v1 * s; kc[o + 1u] = v0 * s + v1 * c;
  }
}`,

  /** One workgroup per query head: causal softmax attention over the cache. */
  attention: /* wgsl */ `
${STEP}
struct P { headSize: u32, kvDim: u32, seqLen: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<f32>;
@group(0) @binding(4) var<storage, read> vc: array<f32>;
@group(0) @binding(5) var<storage, read> headKv: array<u32>;
@group(0) @binding(6) var<storage, read_write> att: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) i: u32) {
  let h = wid.x;
  let hs = p.headSize;
  let n = step.pos + 1u;
  let qo = h * hs;
  let ko = headKv[h] * hs;
  let ao = h * p.seqLen;
  let scale = inverseSqrt(f32(hs));
  var mx = -3.4e38;
  for (var t = i; t < n; t += 64u) {
    var s = 0.0;
    for (var d = 0u; d < hs; d++) { s += q[qo + d] * kc[t * p.kvDim + ko + d]; }
    s *= scale;
    att[ao + t] = s;
    mx = max(mx, s);
  }
  red[i] = mx;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (i < k) { red[i] = max(red[i], red[i + k]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var t = i; t < n; t += 64u) {
    let e = exp(att[ao + t] - m);
    att[ao + t] = e;
    sum += e;
  }
  red[i] = sum;
  storageBarrier();
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (i < k) { red[i] += red[i + k]; }
    workgroupBarrier();
  }
  let total = red[0];
  for (var d = i; d < hs; d += 64u) {
    var acc = 0.0;
    for (var t = 0u; t < n; t++) { acc += att[ao + t] * vc[t * p.kvDim + ko + d]; }
    out[qo + d] = acc / total;
  }
}`,

  swiglu: /* wgsl */ `
${STEP}
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read_write> hb: array<f32>;
@group(0) @binding(3) var<storage, read> hb2: array<f32>;
@group(0) @binding(4) var<storage, read_write> acts: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= p.n) { return; }
  let a = hb[i];
  let v = (a / (1.0 + exp(-a))) * hb2[i];
  hb[i] = v;
  if (step.track == 1u) { acts[i] = acts[i] + abs(v); }
}`,

  /** Adds −log softmax(logits)[target] to loss[0] and 1 to loss[1]. */
  nll: /* wgsl */ `
${STEP}
struct P { vocab: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read> logits: array<f32>;
@group(0) @binding(3) var<storage, read_write> loss: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i: u32) {
  if (step.goal < 0) { return; }
  var mx = -3.4e38;
  for (var j = i; j < p.vocab; j += 256u) { mx = max(mx, logits[j]); }
  red[i] = mx;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (i < k) { red[i] = max(red[i], red[i + k]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = i; j < p.vocab; j += 256u) { sum += exp(logits[j] - m); }
  red[i] = sum;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (i < k) { red[i] += red[i + k]; }
    workgroupBarrier();
  }
  if (i == 0u) {
    loss[0] = loss[0] - (logits[u32(step.goal)] - m - log(red[0]));
    loss[1] = loss[1] + 1.0;
  }
}`,
} as const;

export type KernelName = keyof typeof KERNELS;
