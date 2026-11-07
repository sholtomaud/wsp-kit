/**
 * SentencePiece-style BPE as llama2.c runs it: a vocabulary of pieces with
 * merge scores, byte fallback through the <0xXX> pieces, BOS = 1, EOS = 2.
 */

export const BOS = 1;
export const EOS = 2;

export class Tokenizer {
  readonly pieces: string[];
  readonly scores: Float32Array;
  readonly maxTokenLength: number;
  private readonly lookup = new Map<string, number>();
  /** Token ids of the <0x00>…<0xFF> pieces, or -1 where the vocabulary lacks one. */
  private readonly byteIds = new Int32Array(256).fill(-1);
  private readonly encoder = new TextEncoder();

  constructor(pieces: string[], scores: ArrayLike<number>, maxTokenLength?: number) {
    this.pieces = pieces;
    this.scores = Float32Array.from(scores);
    this.maxTokenLength = maxTokenLength ?? Math.max(1, ...pieces.map((p) => this.encoder.encode(p).length));
    pieces.forEach((p, i) => {
      if (!this.lookup.has(p)) this.lookup.set(p, i);
      const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(p);
      if (m) this.byteIds[parseInt(m[1], 16)] = i;
    });
  }

  get vocabSize() {
    return this.pieces.length;
  }

  /** llama2.c's tokenizer.bin: max_token_length, then (score f32, len i32, bytes) per piece. */
  static fromLlama2c(buffer: ArrayBuffer, vocabSize: number): Tokenizer {
    const view = new DataView(buffer);
    const decoder = new TextDecoder();
    let o = 0;
    const maxTokenLength = view.getInt32(o, true);
    o += 4;
    const pieces: string[] = [];
    const scores = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      if (o + 8 > buffer.byteLength) throw new Error(`Tokenizer file ends at piece ${i} of ${vocabSize}`);
      scores[i] = view.getFloat32(o, true);
      const len = view.getInt32(o + 4, true);
      o += 8;
      pieces.push(decoder.decode(new Uint8Array(buffer, o, len)));
      o += len;
    }
    return new Tokenizer(pieces, scores, maxTokenLength);
  }

  toLlama2c(): ArrayBuffer {
    const bytes = this.pieces.map((p) => this.encoder.encode(p));
    const size = 4 + bytes.reduce((n, b) => n + 8 + b.length, 0);
    const buffer = new ArrayBuffer(size);
    const view = new DataView(buffer);
    view.setInt32(0, this.maxTokenLength, true);
    let o = 4;
    bytes.forEach((b, i) => {
      view.setFloat32(o, this.scores[i], true);
      view.setInt32(o + 4, b.length, true);
      new Uint8Array(buffer, o + 8, b.length).set(b);
      o += 8 + b.length;
    });
    return buffer;
  }

  /** A byte-level vocabulary (<unk>, <s>, </s>, then 256 byte pieces): for models grown from a seed. */
  static byteLevel(): Tokenizer {
    const pieces = ['<unk>', '<s>', '</s>'];
    for (let b = 0; b < 256; b++) pieces.push(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
    return new Tokenizer(pieces, new Float32Array(pieces.length), 6);
  }

  encode(text: string, { bos = true, eos = false } = {}): number[] {
    const tokens: number[] = [];
    if (bos) tokens.push(BOS);
    if (text.length) {
      // SentencePiece's add_dummy_prefix (vocabularies without a " " piece, like
      // the byte-level one, have none).
      const space = this.lookup.get(' ');
      if (space !== undefined) tokens.push(space);
      for (const ch of text) {
        const id = this.lookup.get(ch);
        if (id !== undefined) tokens.push(id);
        else this.pushBytes(ch, tokens);
      }
      // Greedily merge the adjacent pair whose merged piece scores best.
      const start = bos ? 1 : 0;
      for (;;) {
        let best = -Infinity;
        let bestId = -1;
        let bestIdx = -1;
        for (let i = start; i < tokens.length - 1; i++) {
          const id = this.lookup.get(this.pieces[tokens[i]] + this.pieces[tokens[i + 1]]);
          if (id !== undefined && this.scores[id] > best) {
            best = this.scores[id];
            bestId = id;
            bestIdx = i;
          }
        }
        if (bestIdx < 0) break;
        tokens.splice(bestIdx, 2, bestId);
      }
    }
    if (eos) tokens.push(EOS);
    return tokens;
  }

  private pushBytes(ch: string, out: number[]) {
    for (const b of this.encoder.encode(ch)) out.push(this.byteIds[b] >= 0 ? this.byteIds[b] : 0);
  }

  /** The bytes a token stands for, given the token before it. */
  tokenBytes(prev: number, token: number): Uint8Array {
    if (token === BOS || token === EOS || token < 0 || token >= this.pieces.length) return new Uint8Array(0);
    let piece = this.pieces[token];
    const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
    if (m) return Uint8Array.of(parseInt(m[1], 16));
    // SentencePiece strips the dummy prefix after BOS.
    if (prev === BOS && piece.startsWith(' ')) piece = piece.slice(1);
    return this.encoder.encode(piece);
  }

  decode(tokens: number[], prev = -1): string {
    const parts: Uint8Array[] = [];
    for (const t of tokens) {
      parts.push(this.tokenBytes(prev, t));
      prev = t;
    }
    const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) {
      all.set(p, o);
      o += p.length;
    }
    return new TextDecoder().decode(all);
  }

  /** How a token reads, for an LLM: the piece with spaces made visible. */
  describe(token: number): string {
    return JSON.stringify(this.pieces[token] ?? `<${token}?>`);
  }
}
