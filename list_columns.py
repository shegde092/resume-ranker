import pandas as pd, os, sys
path = r'C:\\Users\\shegd\\Documents\\antigravity\\jolly-darwin\\candidates.parquet'
if not os.path.exists(path):
    print('File not found')
    sys.exit(1)
df = pd.read_parquet(path)
print('Columns:', df.columns.tolist())
