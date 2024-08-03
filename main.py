# coding   : utf-8
# @Time    : 2024/7/26
# @Author  : Gscsd
# @File    : main.py
# @Software: PyCharm

import httpx
import numpy as np
from httpx._config import Timeout

from bin.ui import app

# 修正numpy版本兼容问题
np.int = int
old_lst = np.linalg.lstsq
np.linalg.lstsq = lambda a, b, rcond=None: old_lst(a, b, rcond)
# 解决httpx超时问题
httpx._config.DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=30.0)

if __name__ == '__main__':
    app.launch(server_name="0.0.0.0", server_port=5000)
