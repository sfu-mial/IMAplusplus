"""
Constants for the project.
"""

import numpy as np

# Define constants for the source-to-tool mapping.
SOURCE_TOOL_MAPPING = {
    "autofill": "T3",
    np.nan: "T2",
    "manual pointlist": "T1",
}

# Define constants for skill level mapping.
SKILL_LEVEL_MAPPING = {
    "expert": "S1",
    "novice": "S2",
}
