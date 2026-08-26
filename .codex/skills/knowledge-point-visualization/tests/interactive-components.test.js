const fs = require("fs");
const path = require("path");
const vm = require("vm");

const src = fs.readFileSync(path.join(__dirname, "..", "assets", "interactive-components.js"), "utf8");
const sandbox = {
  window: {},
  document: { querySelectorAll: () => [] },
  performance: { now: () => 0 },
  requestAnimationFrame: () => 0,
  cancelAnimationFrame: () => {},
};
sandbox.window.window = sandbox.window;
vm.createContext(sandbox);
vm.runInContext(src, sandbox);

const KPV = sandbox.window.KPV;
if (!KPV || typeof KPV.register !== "function" || typeof KPV.initAll !== "function") {
  throw new Error("KPV API missing");
}
const expected = ["gradient-descent", "projection-box", "implicit-function", "sgld-sampling", "matrix-decomposition"];
for (const id of expected) {
  if (!KPV.demos[id]) throw new Error("missing demo: " + id);
}
const res = KPV.initAll({ querySelectorAll: () => [] });
if (res.total !== 0 || res.ok !== 0 || res.failed.length !== 0) {
  throw new Error("initAll empty case failed: " + JSON.stringify(res));
}
console.log("interactive-components OK: " + expected.join(", "));
