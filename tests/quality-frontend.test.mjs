import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { SLOT_IDS, parseIntake, matchesExpected } from "../dist/quality-intake.mjs";
const html = readFileSync(new URL("../dist/quality.html", import.meta.url), "utf8");
const js = readFileSync(new URL("../dist/quality.mjs", import.meta.url), "utf8");
const readme = readFileSync(new URL("../docs/PRODUCT_NORTH_STAR.md", import.meta.url), "utf8");
test("new quality UI has a usable source/mesh comparison layout", () => {
  for (const marker of ['id="pairs"', 'id="orbits"', 'id="exportManifest"', 'id="meshSha"', 'id="importManifest"', 'quality.mjs', 'quality.css']) {
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

test("manifest handoff accepts eight unique slots but never treats them as restored images", () => {
  const sha = "a".repeat(64);
  const original = {
    format: "echo-visual-review-intake-1",
    scene_id: "desk",
    mesh_sha256: "B".repeat(64),
    notes: "holes remain",
    files: SLOT_IDS.map((slot) => ({ slot, file_name: slot + ".png", sha256: sha }))
  };
  const parsed = parseIntake(original);
  assert.equal(parsed.expected["rgb-1"].sha256, sha);
  assert.equal(parsed.mesh_sha256, "b".repeat(64));
  assert.equal(parsed.notes, "holes remain");
  assert.equal(matchesExpected(sha.toUpperCase(), parsed.expected["rgb-1"].sha256), true);
  assert.equal(matchesExpected("c".repeat(64), parsed.expected["rgb-1"].sha256), false);
  assert.equal(matchesExpected("c".repeat(64), null), true);
  const handoff = structuredClone(original);
  handoff.files[0].sha256 = null;
  handoff.files[0].expected_sha256 = sha;
  assert.equal(parseIntake(handoff).expected["rgb-1"].sha256, sha);
  const duplicate = structuredClone(original);
  duplicate.files[1].slot = "rgb-1";
  assert.throws(() => parseIntake(duplicate), /重复|缺失/);
  const forged = structuredClone(original);
  forged.files[0].sha256 = "not-a-sha";
  assert.throws(() => parseIntake(forged), /SHA-256/);
  assert.throws(() => parseIntake({ ...original, format: "unknown" }), /受支持/);
});
test("quality resume script parses and uses the same local-only manifest helper", () => {
  assert.match(js, /parseIntake\(JSON\.parse\(await file\.text\(\)\)\)/);
  assert.match(js, /matchesExpected\(sha, slot\.expectedSha\)/);
  assert.match(js, /clearSlot\(slot, true\)/);
  execFileSync(process.execPath, ["--check", fileURLToPath(new URL("../dist/quality-intake.mjs", import.meta.url))]);
});
