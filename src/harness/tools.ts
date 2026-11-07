/**
 * The tools the Chief Architect LLM is given. Each maps one-to-one onto a
 * wsp.* method; none of them can move more than a few hundred floats in
 * either direction.
 */
import type { WSP } from '../wsp.ts';
import { S, type Schema } from './schema.ts';

export interface ToolDef {
  name: string;
  description: string;
  input: Schema;
  /** Changes weights: the harness checkpoints first so the change can be undone. */
  mutates?: boolean;
  run(wsp: WSP, input: Record<string, any>): Promise<unknown> | unknown;
}

const target = S.string('Tensor name, e.g. "layer_3.mlp.gate" (see model_overview for the list).');
const texts = (what: string) => S.array(S.string(), what);

export const TOOLS: ToolDef[] = [
  {
    name: 'model_overview',
    description:
      'Architecture of the local network: dims, per-layer heads and MLP widths, and every tensor with its shape and role. Call this first.',
    input: S.object({}),
    run: (wsp) => wsp.inspect.overview(),
  },
  {
    name: 'inspect_summary',
    description:
      'Statistics computed on the GPU: shape, mean, variance, std, min, max, sparsity (fraction of exact zeros), L2 norm, non-finite count. target may be a tensor name, a layer ("layer_2") for all its tensors, or "model".',
    input: S.object({ target: S.string('Tensor name, layer name like "layer_2", or "model".') }, ['target']),
    run: (wsp, i) => wsp.inspect.summary(i.target),
  },
  {
    name: 'inspect_slice',
    description:
      'A Markdown grid of raw values in a block of a tensor. At most 32 rows × 16 columns; bounds are half-open [start, end). Matrices are [rows=outputs, cols=inputs].',
    input: S.object(
      {
        target,
        row_start: S.integer('First row (default 0).'),
        row_end: S.integer('One past the last row (default row_start + 8).'),
        col_start: S.integer('First column (default 0).'),
        col_end: S.integer('One past the last column (default col_start + 8).'),
      },
      ['target']
    ),
    run: (wsp, i) => wsp.inspect.slice(i.target, { rowStart: i.row_start, rowEnd: i.row_end, colStart: i.col_start, colEnd: i.col_end }),
  },
  {
    name: 'inspect_find_anomalies',
    description:
      'Finds mathematically unusual weights with a GPU shader. mode "values": elements whose |z-score| >= threshold, as [row, col, value, z]. mode "rows": rows (MLP neurons in gate/up, tokens in tok_embeddings/lm_head, output units elsewhere) whose L2 norm is an outlier, plus dead (all-zero) rows.',
    input: S.object(
      {
        target,
        threshold: S.number('z-score threshold (default 4).'),
        mode: S.string('"values" (default) or "rows".', ['values', 'rows']),
        limit: S.integer('Max results returned (default 20, max 200).'),
      },
      ['target']
    ),
    run: (wsp, i) => wsp.inspect.findAnomalies(i.target, i.threshold ?? 4, { mode: i.mode, limit: i.limit }),
  },
  {
    name: 'inspect_tokenize',
    description: 'How a text is tokenized: token ids and pieces. Use it to find the rows of tok_embeddings or the classifier that belong to a word.',
    input: S.object({ text: S.string() }, ['text']),
    run: (wsp, i) => wsp.inspect.tokenize(i.text),
  },
  {
    name: 'inspect_neuron',
    description:
      'What one MLP neuron does through the direct path: tokens whose embeddings most activate it, and tokens its output most promotes or suppresses at the classifier (logit lens; ignores the layers in between).',
    input: S.object(
      { layer: S.integer(), neuron: S.integer('Hidden unit index (row of mlp.gate).'), k: S.integer('Tokens per list (default 8).') },
      ['layer', 'neuron']
    ),
    run: (wsp, i) => wsp.inspect.neuron(i.layer, i.neuron, i.k ?? 8),
  },
  {
    name: 'eval_run',
    description:
      'Runs the local network in WebGPU: the greedy continuation of prompt, and if expected_output is given, the loss and perplexity of that exact continuation, plus delta vs. the previous eval of the same pair. Use before and after every mutation to decide keep or rollback.',
    input: S.object(
      {
        prompt: S.string(),
        expected_output: S.string('The continuation you want the model to produce (scored by loss).'),
        max_tokens: S.integer('Tokens to generate (default 48).'),
      },
      ['prompt']
    ),
    run: (wsp, i) => wsp.eval.run(i.prompt, i.expected_output, { maxTokens: i.max_tokens }),
  },
  {
    name: 'eval_next_tokens',
    description: 'The most likely next tokens after a prompt, with probabilities.',
    input: S.object({ prompt: S.string(), k: S.integer('How many (default 8).') }, ['prompt']),
    run: (wsp, i) => wsp.eval.nextTokens(i.prompt, i.k ?? 8),
  },
  {
    name: 'mutate_patch_sparse',
    description:
      'Overwrites (mode "set") or adds to (mode "add") specific floats in VRAM via a scatter shader. Up to 4096 edits per call.',
    mutates: true,
    input: S.object(
      {
        target,
        edits: S.array(S.object({ row: S.integer(), col: S.integer(), value: S.number() }, ['row', 'col', 'value'])),
        mode: S.string('"set" (default) or "add".', ['set', 'add']),
      },
      ['target', 'edits']
    ),
    run: (wsp, i) => wsp.mutate.patchSparse(i.target, i.edits, i.mode ?? 'set'),
  },
  {
    name: 'mutate_graft_node',
    description:
      'Adds capacity to one layer: MLP neurons or attention heads, initialized on the GPU. By default new output weights are zero, so behavior is unchanged until you edit them; init "clone" copies a source neuron/head (plus noise) and with output "split" shares its output (Net2Net, function-preserving). Returns the new indices.',
    mutates: true,
    input: S.object(
      {
        layer: S.integer(),
        kind: S.string(undefined, ['mlp_neurons', 'attention_head']),
        count: S.integer('How many neurons or heads (default 1).'),
        init: S.string('Input weights: "gaussian" (default), "clone", or "zeros".', ['gaussian', 'clone', 'zeros']),
        std: S.number('Std for gaussian init (default: the std of the existing tensor).'),
        source: S.integer('Neuron or head index to clone (init "clone").'),
        noise: S.number('Noise std added to cloned weights (default 1% of the tensor std).'),
        output: S.string('Output weights: "zeros", "gaussian", or "split" (clones only).', ['zeros', 'gaussian', 'split']),
        kv_head: S.integer('Attention heads only: which KV head the new heads read.'),
      },
      ['layer', 'kind']
    ),
    run: (wsp, i) =>
      wsp.mutate.graftNode({
        layer: i.layer,
        kind: i.kind,
        count: i.count,
        init: i.init,
        std: i.std,
        source: i.source,
        noise: i.noise,
        output: i.output,
        kvHead: i.kv_head,
      }),
  },
  {
    name: 'mutate_prune_semantic',
    description:
      'Removes specific knowledge: runs trigger_texts locally, tracks every MLP neuron\'s mean |activation|, and zeroes the neurons most specific to the triggers (gate/up rows and down column). With contrast_texts (knowledge to keep) the score is the trigger/contrast ratio (threshold ~2); without, the z-score within the layer (threshold ~3). Use dry_run first.',
    mutates: true,
    input: S.object(
      {
        trigger_texts: texts('Texts exhibiting the knowledge to remove.'),
        threshold: S.number(),
        contrast_texts: texts('Texts exhibiting knowledge to keep.'),
        layers: S.array(S.integer(), 'Only these layers.'),
        max_neurons: S.integer('At most this many neurons (default 16).'),
        dry_run: S.boolean('Report candidates without changing weights.'),
      },
      ['trigger_texts', 'threshold']
    ),
    run: (wsp, i) =>
      wsp.mutate.pruneSemantic(i.trigger_texts, i.threshold, {
        contrastDataset: i.contrast_texts,
        layers: i.layers,
        maxNeurons: i.max_neurons,
        dryRun: i.dry_run,
      }),
  },
  {
    name: 'mutate_micro_train',
    description:
      'Gradient-free local refinement of ONE tensor (the rest of the network is frozen): a (1+1) evolution strategy that adds seeded noise on the GPU and keeps it only when the loss on texts drops. No backprop. Best on small tensors or a row range.',
    mutates: true,
    input: S.object(
      {
        target,
        texts: texts('Texts whose loss should fall.'),
        steps: S.integer('1–500.'),
        row_start: S.integer('Restrict to rows [row_start, row_end).'),
        row_end: S.integer(),
        sigma: S.number('Initial noise as a fraction of the tensor std (default 0.05).'),
      },
      ['target', 'texts', 'steps']
    ),
    run: (wsp, i) =>
      wsp.mutate.microTrain(i.target, i.texts, i.steps, {
        rows: i.row_start !== undefined && i.row_end !== undefined ? [i.row_start, i.row_end] : undefined,
        sigma: i.sigma,
      }),
  },
  {
    name: 'state_checkpoint',
    description: 'Saves every weight (and the current shape) to the browser\'s private file system under an id.',
    input: S.object({ id: S.string('Letters, digits, . _ -'), note: S.string() }, ['id']),
    run: (wsp, i) => wsp.state.checkpoint(i.id, i.note),
  },
  {
    name: 'state_restore',
    description: 'Rolls the network back to a checkpoint (values and shape). The harness saves "auto-N" before every mutation.',
    input: S.object({ id: S.string() }, ['id']),
    run: (wsp, i) => wsp.state.restore(i.id),
  },
  {
    name: 'state_list',
    description: 'Lists saved checkpoints, newest first.',
    input: S.object({}),
    run: (wsp) => wsp.state.list(),
  },
];

export const TOOL_BY_NAME = new Map(TOOLS.map((t) => [t.name, t]));
