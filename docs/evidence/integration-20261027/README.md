# Integration evidence archive · 2026-10-27

These are byte-for-byte subsets of verified original Actions ZIPs, not regenerated reports. `source-receipts.json` records every included SHA-256 and original run/ZIP identity. Original camera tracks, frozen cohort, measured-depth identities and raw PLY remain available in Git after temporary Actions artifacts expire. Raw public RGB/depth images are not redistributed; original source paths and hashes remain in the reports and official pinned archive.

- `desk-v2-original`: original 40356324356 result, 4543 vertices/7578 faces, coverage 93.7357%, RMSE16.7779cm, FAIL at15cm. Its raw report/Mesh/holdout SHA match the #61 receipt already in main.
- `desk-v2-integration-repeat`: 40490944940, 4572 vertices/7669 faces, coverage94.1953%, RMSE16.5386cm, also FAIL. RGB source-cohort-plan is byte-identical to the original. Fresh COLMAP solving changes the Mesh; these numbers are not an improvement attributable to integration and do not replace the original.

Neither result is independent scanned surface truth or ordinary RGB predicted-depth C evidence. Historical V1 blocked source audit is separately retained at `../desk-v1-original-timestamps-preflight-20261026.json`. No source data, thresholds, units, original reports or Mesh bytes were modified.
