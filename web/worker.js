// The painting worker: loads the same module as the page and hands control to Rust (brushwork-web, worker.rs).
// Requests that arrive while the module is still loading wait in a queue, then go to Rust's handler.
import init, { worker_start } from "./brushwork_web.js";
const waiting = [];
self.onmessage = (e) => waiting.push(e);
init().then(() => {
  worker_start();
  for (const e of waiting) self.onmessage(e);
});
