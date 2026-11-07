import { CpuBackend } from '../src/cpu/backend.ts';
import { seedModel, type SeedOptions } from '../src/model/formats.ts';
import { Model } from '../src/model/model.ts';
import { Tokenizer } from '../src/model/tokenizer.ts';
import { MemoryStore } from '../src/state/store.ts';
import { WSP } from '../src/wsp.ts';

/** A tiny vocabulary with a few merges, enough to exercise BPE. */
export function tinyTokenizer(): Tokenizer {
  const pieces = ['<unk>', '<s>', '</s>'];
  for (let b = 0; b < 256; b++) pieces.push(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
  const words = [' ', 'a', 'b', 'c', 'h', 'e', 'l', 'o', 't', ' c', 'at', ' cat', 'he', 'll', 'hell', 'hello', ' hello'];
  pieces.push(...words);
  const scores = pieces.map((p, i) => (i < 259 ? 0 : p.length * 10 + i / 100));
  return new Tokenizer(pieces, scores);
}

export function makeWsp(seed: SeedOptions = {}) {
  const { config, weights } = seedModel({ dim: 16, layers: 2, heads: 2, kvHeads: 1, hidden: 24, seqLen: 64, std: 0.2, seed: 7, ...seed });
  const tokenizer = Tokenizer.byteLevel();
  const model = new Model(new CpuBackend(), tokenizer, { ...config, vocabSize: tokenizer.vocabSize }, weights, 'test-seed');
  return new WSP(model, { store: new MemoryStore() });
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
