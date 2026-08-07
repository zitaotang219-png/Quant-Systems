from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> int:
    """Seed process-wide random sources used by research components.

    Scikit-learn estimators with ``random_state=None`` draw from NumPy's global
    state, so seeding NumPy also covers that mode. Estimators that expose a
    ``random_state`` parameter should still receive the returned seed.
    """

    resolved_seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(resolved_seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    return resolved_seed
