import { ACTION_DIM, INPUT_DIM, STATE_DIM } from "../yaniv/features";

export interface Net {
  dims: number[];
  weights: Float32Array[];
  biases: Float32Array[];
}

interface LayerSpec {
  in: number;
  out: number;
  w: string;
  b: string;
}

interface NetFile {
  dims: number[];
  layers: LayerSpec[];
  meta?: Record<string, unknown>;
}

function decodeFloat32(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function netFromJson(payload: NetFile): Net {
  if (!payload.layers?.length) throw new Error("model file has no layers");
  const weights = payload.layers.map((layer) => {
    const decoded = decodeFloat32(layer.w);
    if (decoded.length !== layer.in * layer.out) throw new Error("weight size does not match the layer");
    return decoded;
  });
  const biases = payload.layers.map((layer) => {
    const decoded = decodeFloat32(layer.b);
    if (decoded.length !== layer.out) throw new Error("bias size does not match the layer");
    return decoded;
  });
  return { dims: payload.dims, weights, biases };
}

export async function loadNet(url = "/model/yaniv-net.json"): Promise<Net> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`model request failed (${response.status})`);
  return netFromJson((await response.json()) as NetFile);
}

/** y = relu(x @ W + b) for hidden layers, linear for the last. W is row-major (in, out). */
export function forward(net: Net, input: Float32Array): Float32Array {
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

export function scoreActions(net: Net, state: number[], actions: number[][]): Float32Array {
  const scores = new Float32Array(actions.length);
  const packed = new Float32Array(INPUT_DIM);
  for (let row = 0; row < actions.length; row += 1) {
    const action = actions[row];
    if (state.length !== STATE_DIM || action.length !== ACTION_DIM) {
      throw new Error("feature width does not match the network");
    }
    for (let i = 0; i < STATE_DIM; i += 1) packed[i] = state[i];
    for (let i = 0; i < ACTION_DIM; i += 1) packed[STATE_DIM + i] = action[i];
    scores[row] = forward(net, packed)[0];
  }
  return scores;
}
