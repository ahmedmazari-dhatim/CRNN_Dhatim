import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt
import distance

df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/split/digit+char/digit+char.csv', sep=',')
df = df.astype(str)
df = df.replace(['é', 'è', 'È', 'É'], 'e', regex=True)
df = df.replace(['à', 'â', 'Â'], 'a', regex=True)
dictionnary = []
for i in range(len(df)):
    if df.manual_raw_value[i] != df.raw_value[i]:
        text = df.manual_raw_value[i]
        text2 = df.raw_value[i]
        x = len(df.manual_raw_value[i])
        y = len(df.raw_value[i])
        z = min(x, y)
        for t in range(z):
            if text[t] != text2[t]:
                d = (text[t], text2[t])
                dictionnary.append(d)
                print(dictionnary)

#dictionnary_new = collections.Counter(dictionnary).most_common(15)
dictionnary_new = dict(collections.Counter(dictionnary).most_common(15))
pos = np.arange(len(dictionnary_new.keys()))
width = 1.0
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(dictionnary_new.keys())
plt.bar(range(len(dictionnary_new)), dictionnary_new.values(), width, color='g')
plt.show()
