"""Prompt templates for VLM description + condense."""

DESCRIBE_SYSTEM = (
    "You are a careful annotator analyzing video clips of simulated robots in a "
    "physics environment. Your descriptions are used to index policies in a "
    "behavior library, so be concrete, physical, and consistent."
)

DESCRIBE_USER = """\
Below are {n_frames} frames sampled evenly in time order from a {duration:.1f}-second \
clip of a simulated {env_id} robot (one episode/phase).

Describe what the robot is doing. Cover:
  - Body and limb configuration (upright/leaning/inverted, joint posture).
  - Locomotion mode (standing, walking, running, hopping, bounding, falling, recovering, idle).
  - Gait / rhythm (stride length, frequency, symmetry, hop height).
  - Stability and direction of travel.
  - Any notable failure modes (slipping, tumbling, drift).

Avoid speculation about intent. Be specific. 4-6 sentences total.\
"""

CONDENSE_SYSTEM = (
    "You compress detailed robot-motion descriptions into a single short tag-like "
    "phrase suitable for embedding-based retrieval. Output ONLY the phrase, no "
    "preface, no quotes."
)

CONDENSE_USER = """\
Detailed description:
\"\"\"
{detailed}
\"\"\"

Compress this into a single sentence of 10 to 20 words that captures the most \
distinctive behavior. Use concrete physical language (e.g. "fast forward run with \
short symmetric strides", not "the robot moves well"). Output ONLY the sentence.\
"""
