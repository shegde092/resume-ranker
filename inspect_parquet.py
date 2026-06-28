import os, pandas as pd, sys
parquet_path = os.path.join('out', 'candidates.parquet')
if not os.path.exists(parquet_path):
    print('Parquet not found at', parquet_path)
    sys.exit(1)
df = pd.read_parquet(parquet_path)
print('Columns:', df.columns.tolist())
from collections import Counter
cnt = Counter(df['final_tier'])
total = sum(cnt.values())
for tier, c in cnt.items():
    print(tier, c, round(100*c/total, 2))
