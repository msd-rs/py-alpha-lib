---
name: write-algo-integration-testing
description: 为算法实现编写集成测试
---

本技能用于是指导如何为算法编写集成测试.

- 编程语言: Python 3.12+
- 工作目录: 当前工程目录下的 `py-binding`, 以下用 `${WORK_DIR}` 指代
- 对比依赖库: numpy, pandas
- 包管理工具: uv, maturin
- 虚拟环境目录: `${WORK_DIR}/.venv`

# 工作流程

1. 切换目录到 ${WORK_DIR}
  - 检查 Python 环境
  - 安装 pyproject.toml 中的 dev 依赖
  - 运行 `uvx maturin dev -r` 编译本地扩展
2. 获取待编写的集成测试算法列表
  - 通过 `grep "fn ta_" ../alpha-algo/src/algo/*.rs` 获取算法列表, 每个算法函数以 `ta_` 开头, 去掉此前缀即为算法名
  - 通过 `ls itests/test_*.py`, 查看已经完成了集成测试的算法, 每个算法的集成测试名为 `test_${算法名}.py`
  - 对比算法全集和已完成的列表, 从其中挑选一个未编写集成测试的算法, 开始进行编写
3. 编写集成测试
  1. 首先创建 `itests/test_${算法名}.py` 进行占位, 避免冲突
  2. 阅读该算法原 rust 源码, 了解其算法逻辑
  3. 编写等价的基于 numpy/pandas 的实现作为对比基准
    - 允许浮点数在 1e-6 之内的误差
    - 不必实现 SKIP_NAN 机制, 对比实现只处理无 nan 的数据, 但当给数据插入nan后且 alpha 设置了 FLAG_SKIP_NAN, 其结果应该与未插入nan时对比结果一致, 即 FLAG_SKIP_NAN 应确保 NAN 没有参与运算
    - 不必实现 Group 机制, 测试时可以按chunk分块计算
    - 不必实现 start/end 机制, 总是全量计算即可, 但测试时, 应测试 alpha start/end 机制, 仅校验 [start, end) 区间的值即可
    - 对比基准应准确反映算法本身逻辑, 用以校验 alpha 实现的准确性, 不要以 alpha 实现为准, 以算法逻辑为准, 如果某测试不通过, 保留即可, 会有其他的流程进行修复.
  3. 如 `pyproject.toml` 所示, alpha 包位于 `${WORK_DIR}/python` 中, 从这里中导入它,  应使用 alpha.${大写算法名} 作为算法函数, 即经过 python 包装的, 不要直接去使用 `alpha.algo._algo`
  4. 构造不同上下文(alpha.set_ctx)的数据, 进行对比测试
    - Control computation behavior via `alpha.set_ctx()`:
    - **`groups`** — Number of securities in the data array. Each group is processed independently and in parallel. Required for cross-sectional operations like `RANK`.
    - **`start`** — Starting index for calculation (default: 0).
    - **`end`** — Ending index for calculation (default: `len(data)`). `end` can be used when you want to calculate only a part of the data. for example, when back test iteratively.
    - **`flags`** — Bitwise flags:
      - `FLAG_SKIP_NAN` (1): Skip NaN values in rolling windows.
      - `FLAG_STRICTLY_CYCLE` (2): Return NaN until window is full (matches pandas `rolling()` default).
      - Combine with `|`: `flags=FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`
  5. 在文件头部, 以 DocString 格式, 编写相应的说明, 包含算法简介, 不同场景下的数据调用的代码和结果示例