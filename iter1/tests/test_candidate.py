"""Tests for the Candidate dataclass and persistence — ensure curiosity field
doesn't get lost during serialization."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_gen._loop_common import Candidate, save_candidate, pool_dir


class TestCandidate:
    def test_curiosity_default_zero(self):
        c = Candidate(idx=0, source="pass", parent_idx=None, train_seed=0)
        assert c.curiosity == 0.0

    def test_curiosity_in_to_dict(self):
        c = Candidate(idx=0, source="pass", parent_idx=None, train_seed=0, curiosity=4.5)
        d = c.to_dict()
        assert "curiosity" in d
        assert d["curiosity"] == 4.5

    def test_to_dict_roundtrip_json(self):
        c = Candidate(
            idx=7,
            source="def compute_reward(o,a,n,d): return 0.0",
            parent_idx=3,
            train_seed=42,
            curve=[(100, 0.5), (200, 1.0)],
            fitness_vec=[1.0, 2.0, 3.0],
            mean_fitness=2.0,
            descriptor=[0.1, 0.2],
            archive_cell=[1, 2],
            curiosity=5.123,
        )
        d = c.to_dict()
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["curiosity"] == 5.123
        assert d2["mean_fitness"] == 2.0
        assert d2["idx"] == 7

    def test_save_candidate_writes_curiosity(self, tmp_path, monkeypatch):
        """Saved JSON must contain the curiosity field."""
        # Monkey-patch ROOT so pool_dir writes to tmp
        import reward_gen._loop_common as lc
        monkeypatch.setattr(lc, "ROOT", tmp_path)

        c = Candidate(
            idx=0, source="def compute_reward(o,a,n,d): return 0.0",
            parent_idx=None, train_seed=0, curiosity=3.14,
            fitness_vec=[1.0], mean_fitness=1.0,
        )
        save_candidate("pendulum", "curiosity", c)

        json_path = tmp_path / "reward_gen" / "pool" / "pendulum_curiosity" / "reward_000.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["curiosity"] == 3.14


class TestRewardTemplate:
    """Smoke tests for reward compilation to avoid silent breakage."""

    def test_valid_reward_compiles(self):
        from reward_gen.reward_template import compile_reward
        src = "import numpy as np\ndef compute_reward(obs, action, next_obs, done):\n    return float(-np.sum(obs**2))"
        compiled = compile_reward(src)
        assert callable(compiled.fn)

    def test_invalid_import_rejected(self):
        from reward_gen.reward_template import compile_reward, RewardCompileError
        src = "import os\ndef compute_reward(obs, action, next_obs, done):\n    return 0.0"
        with pytest.raises(RewardCompileError, match="disallowed import"):
            compile_reward(src)

    def test_missing_function_rejected(self):
        from reward_gen.reward_template import compile_reward, RewardCompileError
        src = "def some_other_fn(): return 0"
        with pytest.raises(RewardCompileError, match="missing required function"):
            compile_reward(src)

    def test_safe_call_clamps_extreme(self):
        from reward_gen.reward_template import compile_reward, safe_call
        import numpy as np
        src = "def compute_reward(obs, action, next_obs, done):\n    return 1e10"
        compiled = compile_reward(src)
        val = safe_call(compiled, np.zeros(3), np.zeros(1), np.zeros(3), False)
        assert val <= 1e4, "safe_call should clamp extreme values"

    def test_safe_call_handles_nan(self):
        from reward_gen.reward_template import compile_reward, safe_call
        import numpy as np
        src = "import numpy as np\ndef compute_reward(obs, action, next_obs, done):\n    return float('nan')"
        compiled = compile_reward(src)
        val = safe_call(compiled, np.zeros(3), np.zeros(1), np.zeros(3), False)
        assert val == -1e6, "safe_call should return -1e6 for NaN"
