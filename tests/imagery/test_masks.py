import numpy as np

from chla_prediction.imagery.masks import (
    erode_shoreline,
    largest_water_body,
    scl_valid_mask,
    water_extent_mask,
)


def test_scl_valid_mask_excludes_clouds_and_nodata() -> None:
    scl = np.array([[0, 4, 6], [8, 9, 3]], dtype=np.uint8)
    valid = scl_valid_mask(scl)
    assert valid.tolist() == [[False, True, True], [False, False, False]]


def test_erode_shoreline_drops_pixels_touching_land_diagonally() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    # Only the center survives: every other pixel of the 3x3 block touches
    # land, diagonally if not orthogonally.
    assert np.array_equal(erode_shoreline(mask, 1), expected)
    assert not erode_shoreline(mask, 2).any()
    assert np.array_equal(erode_shoreline(mask, 0), mask)


def test_erode_shoreline_treats_the_raster_edge_as_land() -> None:
    mask = np.ones((3, 3), dtype=bool)
    assert np.array_equal(erode_shoreline(mask, 1), np.array([[False] * 3, [False, True, False], [False] * 3]))


def test_largest_water_body_drops_neighboring_bodies() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:6, 1:6] = True  # the water
    mask[7:9, 7:9] = True  # a pond in the same frame
    kept = largest_water_body(mask)
    assert kept[1:6, 1:6].all()
    assert not kept[7:9, 7:9].any()


def test_largest_water_body_ignores_the_frame_center() -> None:
    """The water need not cover the center pixel: on Hushan it does not."""
    mask = np.zeros((9, 9), dtype=bool)
    mask[0:3, 0:4] = True  # off-center water, not touching the center
    mask[4, 4] = True  # an isolated pixel at the frame center
    kept = largest_water_body(mask)
    assert kept[0:3, 0:4].all()
    assert not kept[4, 4]


def test_water_extent_mask_thresholds_then_keeps_the_water_body_then_erodes() -> None:
    raster = np.zeros((9, 9), dtype=np.uint8)
    raster[1:6, 1:6] = 95  # permanent water
    raster[2, 2] = 40  # a gap below threshold, punched into the water
    raster[7:9, 7:9] = 99  # a neighboring body
    mask = water_extent_mask(raster, threshold=90, shoreline_erosion_px=1)
    assert not mask[7:9, 7:9].any()
    assert not mask[1, 1]  # shoreline ring removed
    assert not mask[2, 2]  # sub-threshold pixel and its ring removed
    assert mask[4, 4]
