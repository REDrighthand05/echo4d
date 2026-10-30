// Local-only review handoff. A manifest is a checklist, not verified image evidence.
export const SLOT_IDS = Object.freeze([
  "rgb-1", "projection-1", "rgb-2", "projection-2",
  "rgb-3", "projection-3", "orbit-1", "orbit-2"
]);
const SHA256 = /^[0-9a-f]{64}$/i;
export function parseIntake(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      value.format !== "echo-visual-review-intake-1" || !Array.isArray(value.files) ||
      value.files.length !== SLOT_IDS.length) {
    throw new Error("不是受支持的 ECHO 八图证据清单");
  }
  const seen = new Set();
  const expected = {};
  for (const file of value.files) {
    if (!file || typeof file !== "object" || !SLOT_IDS.includes(file.slot) ||
        seen.has(file.slot) || [file.sha256, file.expected_sha256].some(
          (sha) => sha != null && (typeof sha !== "string" || !SHA256.test(sha)))) {
      throw new Error("证据槽位重复、缺失或 SHA-256 格式错误");
    }
    seen.add(file.slot);
    expected[file.slot] = {
      sha256: (file.sha256 || file.expected_sha256)?.toLowerCase() || null,
      file_name: typeof (file.file_name || file.expected_file_name) === "string"
        ? (file.file_name || file.expected_file_name).slice(0, 256) : null
    };
  }
  if (seen.size !== SLOT_IDS.length) throw new Error("证据槽位不完整");
  if (value.mesh_sha256 != null &&
      (typeof value.mesh_sha256 !== "string" || !SHA256.test(value.mesh_sha256))) {
    throw new Error("Mesh SHA-256 格式错误");
  }
  return {
    scene_id: typeof value.scene_id === "string" ? value.scene_id.slice(0, 100) : "",
    mesh_sha256: value.mesh_sha256?.toLowerCase() || "",
    notes: typeof value.notes === "string" ? value.notes.slice(0, 10000) : "",
    expected
  };
}
export function matchesExpected(actualSha, expectedSha) {
  return expectedSha == null || (typeof actualSha === "string" && actualSha.toLowerCase() === expectedSha);
}
