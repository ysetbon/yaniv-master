import { ACTION_DIM, INPUT_DIM, STATE_DIM } from "../yaniv/features";

/** A plain multilayer perceptron. */
export interface Mlp {
  dims: number[];
  weights: Float32Array[];
  biases: Float32Array[];
}

/**
 * The game's net: a plain one, or a learned decider with expert nets
 * (training/yaniv/moe.py). The decider reads the state and weights the
 * experts; a move's score is the experts' scores mixed by those weights.
 */
export type Net = (Mlp & { kind: "mlp" }) | { kind: "moe"; dims: number[]; gate: Mlp; experts: Mlp[] };

interface LayerSpec {
  in: number;
  out: number;
  w: string;
  b: string;
}

interface NetFile {
  kind?: "moe";
  dims: number[];
  layers?: LayerSpec[];
  gate?: { dims: number[]; layers: LayerSpec[] };
  experts?: { dims: number[]; layers: LayerSpec[] }[];
  meta?: Record<string, unknown>;
}

function decodeFloat32(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function mlpFromLayers(dims: number[], layers: LayerSpec[] | undefined): Mlp {
  if (!layers?.length) throw new Error("model file has no layers");
  const weights = layers.map((layer) => {
    const decoded = decodeFloat32(layer.w);
    if (decoded.length !== layer.in * layer.out) throw new Error("weight size does not match the layer");
    return decoded;
  });
  const biases = layers.map((layer) => {
    const decoded = decodeFloat32(layer.b);
    if (decoded.length !== layer.out) throw new Error("bias size does not match the layer");
    return decoded;
  });
  return { dims, weights, biases };
}

export function netFromJson(payload: NetFile): Net {
  if (payload.kind === "moe") {
    if (!payload.gate || !payload.experts?.length) throw new Error("model file has no decider or experts");
    return {
      kind: "moe",
      dims: payload.dims,
      gate: mlpFromLayers(payload.gate.dims, payload.gate.layers),
      experts: payload.experts.map((expert) => mlpFromLayers(expert.dims, expert.layers)),
    };
  }
  return { kind: "mlp", ...mlpFromLayers(payload.dims, payload.layers) };
}

export async function loadNet(url = "/model/yaniv-net.json"): Promise<Net> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`model request failed (${response.status})`);
  return netFromJson((await response.json()) as NetFile);
}

/** y = relu(x @ W + b) for hidden layers, linear for the last. W is row-major (in, out). */
export function forward(net: Mlp, input: Float32Array): Float32Array {
  let current = input;
  const last = net.weights.length - 1;
  for (let layer = 0; layer <= last; layer += 1) {
    const weight = net.weights[layer];
    const bias = net.biases[layer];
    const widthIn = current.length;
    const widthOut = bias.length;
    const next = new Float32Array(widthOut);
    for (let out = 0; out < widthOut; out += 1) {
      let sum = bias[out];
      for (let inn = 0; inn < widthIn; inn += 1) sum += current[inn] * weight[inn * widthOut + out];
      next[out] = layer === last ? sum : sum > 0 ? sum : 0;
    }
    current = next;
  }
  return current;
}

/** The decider's weight for each expert, from the state alone (softmax of its outputs). */
export function expertWeights(gate: Mlp, state: number[]): number[] {
  const logits = forward(gate, Float32Array.from(state));
  const top = Math.max(...logits);
  const exp = Array.from(logits, (z) => Math.exp(z - top));
  const total = exp.reduce((sum, e) => sum + e, 0);
  return exp.map((e) => e / total);
}

export function scoreActions(net: Net, state: number[], actions: number[][]): Float32Array {
  const scores = new Float32Array(actions.length);
  const packed = new Float32Array(INPUT_DIM);
  const mix = net.kind === "moe" ? expertWeights(net.gate, state) : null;
  for (let row = 0; row < actions.length; row += 1) {
    const action = actions[row];
    if (state.length !== STATE_DIM || action.length !== ACTION_DIM) {
      throw new Error("feature width does not match the network");
    }
    for (let i = 0; i < STATE_DIM; i += 1) packed[i] = state[i];
    for (let i = 0; i < ACTION_DIM; i += 1) packed[STATE_DIM + i] = action[i];
    if (net.kind === "moe" && mix) {
      let score = 0;
      net.experts.forEach((expert, k) => {
        score += mix[k] * forward(expert, packed)[0];
      });
      scores[row] = score;
    } else if (net.kind === "mlp") {
      scores[row] = forward(net, packed)[0];
    }
  }
  return scores;
}
