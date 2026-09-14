"""`save_view` is what makes an observation something an agent can look at."""

import numpy as np
import pytest

from harness.perception import save_view

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture
def obs():
    rng = np.random.default_rng(0)
    return {
        "robot0_eye_in_hand_image": rng.integers(0, 256, (8, 12, 3), dtype=np.uint8),
        "robot0_eye_in_hand_depth": rng.random((8, 12)).astype(np.float32),
        "robot0_joint_vel": np.zeros(7),
    }


def test_images_survive_the_round_trip(tmp_path, obs):
    written = save_view(obs, str(tmp_path))
    got = np.asarray(PIL.open(written["robot0_eye_in_hand_image"]))
    assert np.array_equal(got, obs["robot0_eye_in_hand_image"])


def test_depth_stays_metres_rather_than_a_picture(tmp_path, obs):
    """Range is measured, not looked at: normalising it into a PNG would destroy the one
    thing it is for."""
    written = save_view(obs, str(tmp_path))
    assert written["robot0_eye_in_hand_depth"].endswith(".npy")
    assert np.array_equal(np.load(written["robot0_eye_in_hand_depth"]),
                          obs["robot0_eye_in_hand_depth"])


def test_proprioception_is_not_written(tmp_path, obs):
    assert "robot0_joint_vel" not in save_view(obs, str(tmp_path))
