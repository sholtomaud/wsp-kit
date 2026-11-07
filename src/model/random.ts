/**
 * Counter-based randomness, identical in TypeScript and WGSL: value i of a
 * stream depends only on (seed, i), so a GPU fill and a CPU fill agree and a
 * mutation can be replayed from its seed.
 */

export function pcg(v: number): number {
  const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul(((state >>> (((state >>> 28) + 4) >>> 0)) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

const f32 = Math.fround;

/** A standard normal sample (Box–Muller), computed in f32 steps like the shader. */
export function gaussian(seed: number, i: number): number {
  const h1 = pcg((i ^ pcg(seed >>> 0)) >>> 0);
  const h2 = pcg(h1);
  const u1 = f32(f32(f32(h1) + 0.5) / 4294967296);
  const u2 = f32(f32(f32(h2) + 0.5) / 4294967296);
  return f32(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
}

export const PCG_WGSL = /* wgsl */ `
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn gaussian(seed: u32, i: u32) -> f32 {
  let h1 = pcg(i ^ pcg(seed));
  let h2 = pcg(h1);
  let u1 = (f32(h1) + 0.5) / 4294967296.0;
  let u2 = (f32(h2) + 0.5) / 4294967296.0;
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
`;

/** mulberry32, for sampling tokens on the CPU. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
