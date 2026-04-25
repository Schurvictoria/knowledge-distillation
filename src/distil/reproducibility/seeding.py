"""
Одна функция чтобы засидить ВСЕ источники случайности в Python ML стеке.

Заменяет 8-строчный блок что был в каждом скрипте:
  random.seed + np.random.seed + torch.manual_seed + torch.cuda.manual_seed_all +
  pl.seed_everything(workers=True) + PYTHONHASHSEED + cudnn.deterministic + cudnn.benchmark.

torch и pytorch_lightning импортируются опционально — если не установлены,
функция просто пропустит их сидирование (полезно для CPU-only OpenRouter скриптов).
"""
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass
