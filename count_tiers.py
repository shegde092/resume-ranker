import pandas as pd
from collections import Counter

df = pd.read_parquet(r'C:/Users/shegd/Documents/antigravity/jolly-darwin/candidates.parquet')
print('Columns:', list(df.columns))
cnt = Counter(df['final_tier'])
total = sum(cnt.values())
for tier, count in cnt.items():
    print(tier, count, round(100*count/total, 2))
