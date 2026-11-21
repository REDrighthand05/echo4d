"""Unit tests for procedural PBR material library and texture generation."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
import sys
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from echo_scene.materials import (
    ADVANCED_PRESETS,
    INDOOR_PRESETS,
    MaterialDescriptor,
    MaterialFamily,
    bake_pbr_textures,
    cellular_worley_noise,
    crack_network_pattern,
    directional_noise,
    fbm_noise,
    get_material_preset,
    hexagonal_honeycomb_pattern,
    list_all_presets,
    twill_weave_pattern,
)
from echo_scene.materials.pbr_maps import (
    height_to_normal_map,
    pack_metallic_roughness_map,
)
from echo_scene.materials.presets import synthesize_material_textures
from echo_scene.materials.procedural import linear_to_srgb, srgb_to_linear


class TestProceduralMaterials(unittest.TestCase):
    def test_catalog_coverage(self):
        all_presets = list_all_presets()
        self.assertEqual(len(all_presets), 16)
        self.assertEqual(len(INDOOR_PRESETS), 8)
        self.assertEqual(len(ADVANCED_PRESETS), 8)

        for name in all_presets:
            mat = get_material_preset(name)
            self.assertIsInstance(mat, MaterialDescriptor)
            self.assertEqual(mat.id, name)
            d = mat.to_dict()
            reconstructed = MaterialDescriptor.from_dict(d)
            self.assertEqual(reconstructed.id, mat.id)
            self.assertEqual(reconstructed.family, mat.family)

    def test_color_space_conversions(self):
        vals = np.array([0.0, 0.04045, 0.5, 1.0], dtype=np.float32)
        lin = srgb_to_linear(vals)
        roundtrip = linear_to_srgb(lin)
        np.testing.assert_allclose(vals, roundtrip, atol=1e-5)

    def test_patterns(self):
        shape = (64, 64)
        fbm = fbm_noise(shape, octaves=3)
        self.assertEqual(fbm.shape, shape)
        self.assertGreaterEqual(fbm.min(), 0.0)
        self.assertLessEqual(fbm.max(), 1.0)

        d_noise = directional_noise(shape, angle_rad=0.5)
        self.assertEqual(d_noise.shape, shape)

        worley = cellular_worley_noise(shape, num_cells=16)
        self.assertEqual(worley.shape, shape)

        twill_h, twill_dir = twill_weave_pattern(shape, yarn_count=4)
        self.assertEqual(twill_h.shape, shape)
        self.assertEqual(twill_dir.shape, shape)

        honeycomb = hexagonal_honeycomb_pattern(shape, cell_freq=4.0)
        self.assertEqual(honeycomb.shape, shape)

        cracks = crack_network_pattern(shape, num_seeds=10)
        self.assertEqual(cracks.shape, shape)

    def test_normal_map_computation(self):
        # Flat surface height = 0.5 -> normal should be flat [128, 128, 255]
        flat_height = np.full((32, 32), 0.5, dtype=np.float32)
        normal = height_to_normal_map(flat_height, normal_strength=1.0)
        self.assertEqual(normal.shape, (32, 32, 3))
        # Center pixels should be exactly [128, 128, 255]
        np.testing.assert_allclose(normal[10:20, 10:20, :2], 128, atol=2)
        np.testing.assert_allclose(normal[10:20, 10:20, 2], 255, atol=2)

    def test_pack_metallic_roughness(self):
        rough = np.full((16, 16), 0.75, dtype=np.float32)
        metal = np.full((16, 16), 0.25, dtype=np.float32)
        orm = pack_metallic_roughness_map(rough, metal)
        self.assertEqual(orm.shape, (16, 16, 3))
        self.assertEqual(orm[0, 0, 0], 255)  # R: Occlusion default
        self.assertAlmostEqual(orm[0, 0, 1], int(0.75 * 255), delta=1)  # G: Roughness
        self.assertAlmostEqual(orm[0, 0, 2], int(0.25 * 255), delta=1)  # B: Metallic

    def test_texture_synthesis_and_baking(self):
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td)
            # Test painted wall
            mat_wall = get_material_preset("painted_wall")
            synthesize_material_textures(mat_wall, out_p, res=64)
            self.assertIsNotNone(mat_wall.textures.base_color_map)
            self.assertTrue(Path(mat_wall.textures.base_color_map).exists())
            self.assertTrue(Path(mat_wall.textures.normal_map).exists())
            self.assertTrue(Path(mat_wall.textures.metallic_roughness_map).exists())

            # Test advanced brushed metal
            mat_brushed = get_material_preset("brushed_metal")
            synthesize_material_textures(mat_brushed, out_p, res=64)
            self.assertTrue(Path(mat_brushed.textures.normal_map).exists())

            # Test carbon fiber twill
            mat_cf = get_material_preset("carbon_fiber_twill")
            synthesize_material_textures(mat_cf, out_p, res=64)
            self.assertTrue(Path(mat_cf.textures.base_color_map).exists())


if __name__ == "__main__":
    unittest.main()
