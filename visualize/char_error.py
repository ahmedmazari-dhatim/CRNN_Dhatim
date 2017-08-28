# in the same plot make error crnn and error abby
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import distance
# crnn error and abby error
input_file ='/home/ahmed/Pictures/cogedis/cogedis_words_3/words.csv'




def histo_error(input_path):
    df = pd.read_csv(input_path, sep=',')
    print(len(df))
    df = df.astype(str)
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

    dictionnary_new = collections.Counter(dictionnary)
    key = []
    value = []

    for k, v in dictionnary_new.most_common(15):
        key.append(k)
        value.append(v)

    pos = np.arange(len(key))
    width = 0.25

    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(key)

    plt.bar(range(len(key)), value, width, color='g')
    plt.xticks(rotation=90)
    plt.show()
    plt.show()

def levensetein_error(input_path):
    df = pd.read_csv(input_path, sep=',')
    print(len(df))
    df = df.astype(str)
    df['string diff'] = df.apply(lambda x: distance.levenshtein(x['manual_raw_value'], x['raw_value']), axis=1)
    plt.hist(df['string diff'])
    plt.show()



if __name__ == "__main__":
    #histo_error(input_file)
    levensetein_error(input_file)
