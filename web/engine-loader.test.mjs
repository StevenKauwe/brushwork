import {test} from "node:test";
import assert from "node:assert/strict";
import {boot, usableBuilds} from "./engine-loader.mjs";
test("real SIMD probe and unsupported/throwing hosts", () => {
  assert.deepEqual(usableBuilds(), ["simd", "scalar"]);
  assert.deepEqual(usableBuilds(() => false), ["scalar"]);
  assert.deepEqual(usableBuilds(() => {throw Error("disabled")}), ["scalar"]);
});
test("SIMD init failure retries scalar and reports its actual engine", async () => {
  const calls=[];
  const engine=await boot(async ({module_or_path}) => {calls.push(module_or_path.pathname);if(calls.length===1)throw Error("unsupported");}, {validate:()=>true});
  assert.equal(engine,"scalar");assert.equal(calls.length,2);assert.match(calls[1],/bg.wasm$/);
});
test("forced engine is honored, unsupported requests and total failures reject", async () => {
  assert.equal(await boot(async()=>{}, {engine:"scalar"}),"scalar");
  await assert.rejects(boot(async()=>{}, {engine:"simd",validate:()=>false}), /does not support/);
  await assert.rejects(boot(async()=>{}, {engine:"surprise"}), /Unknown/);
  await assert.rejects(boot(async()=>{throw Error("broken")}), /simd: broken; scalar: broken/);
});
