import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
const html = readFileSync(new URL("../dist/quality.html", import.meta.url), "utf8");
const js = readFileSync(new URL("../dist/quality.mjs", import.meta.url), "utf8");
const readme = readFileSync(new URL("../docs/PRODUCT_NORTH_STAR.md", import.meta.url), "utf8");
test("new quality UI has a usable source/mesh comparison layout", () => {
  for (const marker of ['id="pairs"', 'id="orbits"', 'id="exportManifest"', 'id="meshSha"', 'quality.mjs', 'quality.css']) {
    assert.ok(html.includes(marker), marker);
  }
  assert.match(js, /crypto\.subtle\.digest\("SHA-256"/);
  assert.match(js, /overall_accepted: false/);
  assert.match(js, /source_verified: false/);
  assert.match(readme, /42条/);
});
test("quality script parses on Node 22 without imports or network dependencies", () => {
  execFileSync(process.execPath, ["--check", fileURLToPath(new URL("../dist/quality.mjs", import.meta.url))]);
});
