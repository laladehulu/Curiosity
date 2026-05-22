"""Per-task perturbation sets for Phase 4. Per protocol §"Perturbation protocol".

Each entry returns a (env_id, kwargs_or_patcher) that produces a perturbed env.
We perturb via gymnasium env attributes after instantiation since not all envs
take kwargs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym


@dataclass
class Perturbation:
    name: str
    apply: Callable[[gym.Env], gym.Env]  # mutate env in-place or wrap


def _pendulum_grav(g_mult: float):
    def apply(env: gym.Env) -> gym.Env:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "g"):
            unwrapped.g = 10.0 * g_mult
        return env
    return apply


def _pendulum_mass(m_mult: float):
    def apply(env: gym.Env) -> gym.Env:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "m"):
            unwrapped.m = 1.0 * m_mult
        return env
    return apply


def _pendulum_length(l_mult: float):
    def apply(env: gym.Env) -> gym.Env:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "l"):
            unwrapped.l = 1.0 * l_mult
        return env
    return apply


def _mountaincar_power(p_mult: float):
    def apply(env: gym.Env) -> gym.Env:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "power"):
            unwrapped.power = 0.0015 * p_mult
        return env
    return apply


def _mountaincar_gravity(g_mult: float):
    def apply(env: gym.Env) -> gym.Env:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "gravity"):
            unwrapped.gravity = 0.0025 * g_mult
        return env
    return apply


def _bipedal_grav(g_mult: float):
    def apply(env: gym.Env) -> gym.Env:
        # Box2D world gravity
        if hasattr(env.unwrapped, "world"):
            env.unwrapped.world.gravity = (0.0, -10.0 * g_mult)
        return env
    return apply


def _bipedal_friction(f_mult: float):
    """Friction perturbation: scale the friction joint coefficient. Best-effort —
    Box2D fixtures are created on reset, so this must be applied after each reset."""
    def apply(env: gym.Env) -> gym.Env:
        unwrapped = env.unwrapped
        # most reliable hook is to scale post-reset; we tag the env so wrapper can re-apply
        unwrapped._friction_mult = f_mult  # consumed below if needed
        return env
    return apply


PERTURBATIONS: dict[str, list[Perturbation]] = {
    "pendulum": [
        Perturbation("grav_-30%", _pendulum_grav(0.70)),
        Perturbation("grav_+30%", _pendulum_grav(1.30)),
        Perturbation("mass_-20%", _pendulum_mass(0.80)),
        Perturbation("mass_+20%", _pendulum_mass(1.20)),
        Perturbation("len_-20%",  _pendulum_length(0.80)),
        Perturbation("len_+20%",  _pendulum_length(1.20)),
    ],
    "mountaincar": [
        Perturbation("grav_-30%",  _mountaincar_gravity(0.70)),
        Perturbation("grav_+30%",  _mountaincar_gravity(1.30)),
        Perturbation("power_-30%", _mountaincar_power(0.70)),
        Perturbation("power_+30%", _mountaincar_power(1.30)),
    ],
    "bipedalwalker": [
        Perturbation("grav_-20%", _bipedal_grav(0.80)),
        Perturbation("grav_+20%", _bipedal_grav(1.20)),
        Perturbation("friction_-30%", _bipedal_friction(0.70)),
        Perturbation("friction_+30%", _bipedal_friction(1.30)),
        Perturbation("terrain_seed_alt", lambda env: env),  # alternative-seed handled at reset time
    ],
}


def task_success_threshold(task: str) -> float:
    """Success criterion per task for "did this policy succeed on this perturbed env"."""
    return {
        "pendulum": -200.0,         # raw env return; standard "solved" is > -200
        "mountaincar": 90.0,        # +100 on goal minus action cost
        "bipedalwalker": 100.0,     # forward progress; partial success threshold
    }[task]
