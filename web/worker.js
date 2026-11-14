// The painting worker: loads the same module as the page and hands control to Rust (brushwork-web, worker.rs).
// Requests that arrive while the module is still loading wait in a queue, then go to Rust's handler.
import init, { worker_start } from "./brushwork_web.js";
import { boot } from "./engine-loader.mjs";
const waiting = [];
self.onmessage = (e) => waiting.push(e);
boot(init, {engine: new URL(import.meta.url).searchParams.get("engine")}).then(() => {
  worker_start();
  for (const e of waiting) self.onmessage(e);
}).catch(error => self.postMessage(["backend", "cpu", "Painter failed to start: " + error.message]));
