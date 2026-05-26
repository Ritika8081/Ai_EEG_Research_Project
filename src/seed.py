import os
import random
import numpy as np
import torch


def seed_all(seed: int = 1337) -> None:
    # Fixed seed everywhere. Documented in README. Deterministic algorithms
    # are intentionally NOT forced — cuDNN determinism roughly doubles runtime
    # and these experiments are small enough that within-seed variance is the
    # dominant concern, not run-to-run drift from non-deterministic kernels.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
