/**
 * wsp-kit: browser-native Weight-Space Programming.
 *
 *   const wsp = await createWsp(await loadStarter('stories260K'));
 *   await wsp.inspect.summary('layer_3.mlp.gate');
 *   const harness = new Harness({ wsp, provider: () => new AnthropicProvider({ apiKey }) });
 *   await harness.send('The model repeats itself. Find out why and fix it.');
 */
export * from './backend.ts';
export { CpuBackend } from './cpu/backend.ts';
export { WebGpuBackend, requestDevice, WgslError } from './gpu/backend.ts';
export * from './model/types.ts';
export * from './model/formats.ts';
export { Tokenizer, BOS, EOS } from './model/tokenizer.ts';
export { Model, type GenerateOptions } from './model/model.ts';
export * from './state/store.ts';
export * from './state/datasets.ts';
export * from './wsp.ts';
export * from './harness/schema.ts';
export * from './harness/tools.ts';
export * from './harness/agent.ts';
export * from './llm/types.ts';
export * from './llm/anthropic.ts';
export * from './llm/openai-compatible.ts';
export * from './llm/gemini.ts';
export * from './llm/keys.ts';
export * from './loader.ts';
