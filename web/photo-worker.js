import init, { paint_photo_rgba } from "./brushwork_web.js";
import { boot } from "./engine-loader.mjs";
const ready = boot(init, {engine: new URL(import.meta.url).searchParams.get("engine")}).then(engine=>({engine}), error=>({error}));
let queue = Promise.resolve();
self.onmessage = ({data}) => {
  // One WASM instance/painting at a time. New jobs never interleave GPU submissions.
  queue = queue.then(async () => {
    const loaded = await ready;
    if (loaded.error) throw loaded.error;
    const engine = loaded.engine;
    const result = await paint_photo_rgba(data.pixels, data.width, data.height, data.resolution ?? 540, data.seed ?? 1, !!data.steady, !!data.cpu);
    self.postMessage({id: data.id, engine, ...result}, [result.pixels.buffer]);
  }).catch(error => self.postMessage({id: data.id, error: String(error)}));
};
