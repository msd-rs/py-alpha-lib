import pandas as pd
import numpy as np
import alpha


df = pd.DataFrame(
  data={
    "Animal": ["cat", "penguin", "dog", "spider", "snake"],
    "Number_legs": [4, 2, 4, 8, 0],
  }
)


alpha.set_ctx(groups=5)
print(alpha.RANK(df["Number_legs"].to_numpy().astype(np.float64)))
print(df["Number_legs"].rank(pct=True).to_numpy().astype(np.float64))
