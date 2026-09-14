"""Tests for the client<->daemon wire protocol.

The agent->daemon direction crosses into the trusted side, so action validation is a
security surface, not a convenience: malformed input must be refused rather than
repaired, and nothing may be unpickled.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import protocol as P


def roundtrip(obj):
    import json

    return P.decode(json.loads(json.dumps(P.encode(obj))))


def test_roundtrips_nested_observation_with_images():
    obs = {
        "robot0_eef_pos": np.array([1.0, -2.5, 0.3]),
        "robot0_agentview_left_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "object-state": np.arange(6, dtype=np.float32),
        "meta": {"lang": "Close the left drawer.", "step": 3},
    }
    out = roundtrip(obs)
    assert np.allclose(out["robot0_eef_pos"], obs["robot0_eef_pos"])
    assert out["robot0_agentview_left_image"].shape == (8, 8, 3)
    assert out["robot0_agentview_left_image"].dtype == np.uint8
    assert out["object-state"].dtype == np.float32
    assert out["meta"]["lang"] == "Close the left drawer."


def test_roundtrip_preserves_uint8_image_content():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    assert np.array_equal(roundtrip({"i": img})["i"], img)


def test_numpy_scalars_become_plain_python():
    out = roundtrip({"a": np.float32(1.5), "b": np.int64(7)})
    assert out == {"a": pytest.approx(1.5), "b": 7}


def test_object_dtype_arrays_are_refused():
    """An object array is the classic vector for smuggling arbitrary payloads."""
    payload = {P._ARRAY_TAG: True, "dtype": "object", "shape": [2], "data": ""}
    with pytest.raises(P.ProtocolError, match="dtype"):
        P.decode(payload)


def test_shape_inconsistent_array_is_refused():
    import base64

    payload = {
        P._ARRAY_TAG: True,
        "dtype": "float64",
        "shape": [100],
        "data": base64.b64encode(np.zeros(2).tobytes()).decode(),
    }
    with pytest.raises(P.ProtocolError, match="elements"):
        P.decode(payload)


# -- action validation ------------------------------------------------------

def test_valid_action_passes_through():
    a = P.validate_action([0.0] * 12, expected_dim=12)
    assert a.shape == (12,)


def test_action_is_clipped_to_the_documented_interval():
    """Clipping is the contract: every RoboCasa action component is on [-1, 1]."""
    a = P.validate_action([5.0] * 12, expected_dim=12)
    assert np.all(a == 1.0)
    a = P.validate_action([-5.0] * 12, expected_dim=12)
    assert np.all(a == -1.0)


def test_wrong_length_action_is_refused_not_padded():
    with pytest.raises(P.ProtocolError, match="12 components"):
        P.validate_action([0.0] * 7, expected_dim=12)


def test_nan_and_inf_actions_are_refused():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(P.ProtocolError, match="NaN or infinity"):
            P.validate_action([bad] + [0.0] * 11, expected_dim=12)


def test_non_1d_action_is_refused():
    with pytest.raises(P.ProtocolError, match="1-D"):
        P.validate_action(np.zeros((2, 12)), expected_dim=12)


def test_non_numeric_action_is_refused():
    with pytest.raises((P.ProtocolError, ValueError, TypeError)):
        P.validate_action(["nope"] * 12, expected_dim=12)


# -- framing ----------------------------------------------------------------

def test_line_reader_reassembles_messages_split_across_chunks():
    import json

    class ChunkSock:
        def __init__(self, blob, size):
            self.blob, self.size, self.pos = blob, size, 0

        def recv(self, _n):
            chunk = self.blob[self.pos: self.pos + self.size]
            self.pos += len(chunk)
            return chunk

    msgs = [{"op": "step", "i": i} for i in range(3)]
    blob = b"".join(
        json.dumps(P.encode(m), separators=(",", ":")).encode() + b"\n" for m in msgs
    )
    reader = P.LineReader(ChunkSock(blob, size=3))  # tiny chunks, many partial reads
    assert [reader.read() for _ in msgs] == msgs
    assert reader.read() is None  # closed


def test_line_reader_rejects_a_message_that_never_terminates():
    class FloodSock:
        def recv(self, _n):
            return b"x" * 4096

    reader = P.LineReader(FloodSock(), max_bytes=8192)
    with pytest.raises(P.ProtocolError, match="size limit"):
        reader.read()
