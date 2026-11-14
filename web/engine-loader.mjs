// Capability-first loading, following BrowserTabICL's independently built WASM variants.
// No relaxed-SIMD instructions are shipped: plain SIMD and a portable scalar fallback.
export const builds = {simd: "brushwork_web_simd.wasm", scalar: "brushwork_web_bg.wasm"};
const simdProbe = new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,23,1,21,0,253,12,...Array(16).fill(0),26,11]);
export function usableBuilds(validate = bytes => WebAssembly.validate(bytes)) {
  try { if (validate(simdProbe)) return ["simd", "scalar"]; } catch {}
  return ["scalar"];
}
export async function boot(init, {engine, validate} = {}) {
  const candidates = usableBuilds(validate);
  if (engine && !Object.hasOwn(builds, engine)) throw new Error("Unknown WASM engine: " + engine);
  if (engine && !candidates.includes(engine)) throw new Error("This browser does not support " + engine);
  const failures = [];
  for (const name of engine ? [engine] : candidates) {
    try {
      await init({module_or_path: new URL(builds[name], import.meta.url)});
      return name;
    } catch (error) { failures.push(name + ": " + error.message); }
  }
  throw new Error("Could not start the painter. " + failures.join("; "));
}
