import os, sys, pandas as pd
parquet_path = r'C:/Users/shegd/Documents/antigravity/jolly-darwin/candidates.parquet'
if not os.path.exists(parquet_path):
    print('Parquet missing at', parquet_path)
    sys.exit(1)
df = pd.read_parquet(parquet_path)
print('Columns:', df.columns.tolist())
from collections import Counter
cnt = Counter(df['final_tier'])
total = sum(cnt.values())
print('Tier distribution:')
for tier, c in cnt.items():
    print(tier, c, round(100 * c / total, 2))
