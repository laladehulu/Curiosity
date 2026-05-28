"""Prompts for reward generation + mutation (Eureka-style)."""

SYSTEM = """\
You are an expert reinforcement-learning engineer designing reward functions
for the Gymnasium Hopper-v5 environment (MuJoCo, 1D forward locomotion of a
one-legged hopper).

Observation layout (shape (11,)):
  obs[0]  z-coordinate of the top (body height)
  obs[1]  angle of the top
  obs[2]  thigh joint angle
  obs[3]  leg joint angle
  obs[4]  foot joint angle
  obs[5]  x-velocity (forward velocity of the body)   <-- positive = forward
  obs[6]  z-velocity (vertical velocity of the body)
  obs[7]  angular velocity of the top
  obs[8]  angular velocity of thigh
  obs[9]  angular velocity of leg
  obs[10] angular velocity of foot

Action layout (shape (3,), each in [-1, 1]):
  action[0]  thigh torque
  action[1]  leg torque
  action[2]  foot torque

Your job is to write a Python function:

    def compute_reward(obs, action, next_obs, done):
        # obs, action, next_obs are numpy arrays; done is bool
        # return a finite scalar
        return r

Constraints (will be enforced; violating them just makes your reward less
useful, not catastrophic):
  - No imports. `np` and `math` are pre-bound.
  - No attribute access starting with `_`.
  - No exec/eval/open/file IO.
  - Return value will be clamped to [-100, 100] and finite.

Design philosophy: think about what *behavior* your reward shapes, not just
what numbers it produces. Hopper succeeds when the agent moves forward
without falling over; many distinct gaits achieve this. Your reward will
shape which one emerges.
"""

COLD_GEN = """\
Propose a `compute_reward` function for Hopper-v5. Output ONLY a Python
code block (triple-backticks) containing the function definition — no
other text, no explanation.
"""

MUTATE_PARENT_ONLY = """\
The reward function below produced a Hopper policy with fitness {parent_fit:.2f}
(under the default Hopper reward) and the following observed behavior:

  "{parent_desc}"

Parent reward:
```python
{parent_code}
```

Propose a *variant* of this reward function. Keep the parts that seem to
help and modify the parts that seem to limit the behavior. Output ONLY a
Python code block containing the new compute_reward — no other text.
"""

MUTATE_WITH_NEIGHBORS = """\
The reward function below produced a Hopper policy with fitness {parent_fit:.2f}
and behavior:

  "{parent_desc}"

Parent reward:
```python
{parent_code}
```

Nearby (similar-behavior) policies in our archive include:
{neighbor_list}

Propose a *variant* of the parent reward whose resulting policy behaves
differently from these neighbors. Push the gait toward something not
already covered — for example a different stride length, posture, speed
range, or stability profile. Output ONLY a Python code block containing
the new compute_reward — no other text.
"""


def format_neighbor_list(neighbors: list[tuple[str, float, str]]) -> str:
    """neighbors = [(short_id, fitness, description), ...]"""
    if not neighbors:
        return "  (none — this is the only entry in its region)"
    lines = []
    for short_id, fit, desc in neighbors:
        lines.append(f'  - [{short_id}, fit={fit:.2f}] "{desc}"')
    return "\n".join(lines)
