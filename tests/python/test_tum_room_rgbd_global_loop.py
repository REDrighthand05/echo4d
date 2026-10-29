"""CPU regressions for nonlocal visual retrieval and verified RGB-D loop edges."""
import cv2
import numpy as np
import pytest

pytest.importorskip("open3d")
from tum_room_rgbd_global_loop import global_candidates, connected_sizes, checked_loop_edge


def test_bow_retrieves_distant_repeated_view_without_gt_poses():
    rng = np.random.default_rng(42)
    common = rng.normal(128, 15, (35, 128)).astype(np.float32)
    other = [rng.normal(20 + i * 7, 4, (35, 128)).astype(np.float32)
             for i in range(12)]
    features = [(None, d) for d in other]
    features[0] = (None, common)
    features[11] = (None, common.copy())
    pairs, report = global_candidates(features, 12, exclude_gap=3, per_frame=4, clusters=12)
    assert (0, 11) in pairs
    assert report["proposed_nonlocal_pairs"] > 0


def test_connected_components_are_real_graph_edges_only():
    T = np.eye(4)
    graph = [[(1, T, 1.0)], [(0, T, 1.0)], [], [(4, T, 1.0)], [(3, T, 1.0)]]
    assert connected_sizes(graph) == [2, 2, 1]


def test_global_loop_edge_requires_observed_depth_supported_pnp():
    rng = np.random.default_rng(21)
    image = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    shifted = np.zeros_like(image)
    shifted[:, :-16] = image[:, 16:]
    sift = cv2.SIFT_create(nfeatures=3000)
    features = [sift.detectAndCompute(im, None) for im in (image, shifted)]
    z = np.full((240, 320), 2.0, dtype=np.float32)
    K = np.array([[160., 0, 159.5], [0, 160., 119.5], [0, 0, 1.]])
    edge = checked_loop_edge(features, [z, z], K, 0, 1, cv2.BFMatcher(cv2.NORM_L2))
    assert edge is not None
    assert abs(edge[0][0, 3]) > .05
    missing = np.zeros_like(z)
    assert checked_loop_edge(features, [missing, missing], K, 0, 1,
                             cv2.BFMatcher(cv2.NORM_L2)) is None
