
df = pd.read_csv(filename, sep=",",index_col=None, error_bad_lines=False)
df.columns = range(0, df.shape[1])
for i in range(4, df.shape[1]):
    df = df[df.iloc[:,i].isnull()]
df = df[[0, 1, 2, 3]]
df.rename(columns={'0': 'id', '1': 'ocr','2':'raw_value','3':'manual_raw_value'}, inplace=True)

