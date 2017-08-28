from __future__ import division
import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt
import distance
from subs_del_inser import levenshtein_ids
import pylab as plt




def error_dist():
    df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/processed.csv', sep=',')
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
                    #print(dictionnary)

    dictionnary_new = dict(collections.Counter(dictionnary).most_common(25))

    pos = np.arange(len(dictionnary_new.keys()))
    width = 1.0

    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(dictionnary_new.keys())

    plt.bar(range(len(dictionnary_new)), dictionnary_new.values(), width, color='g')

    plt.show()


def levenstein_dist():
    df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/processed.csv', sep=',')
    df=df.astype(str)
    df['string diff'] = df.apply(lambda x: distance.levenshtein(x['raw_value'], x['manual_raw_value']), axis=1)
    plt.hist(df['string diff'])
    plt.show()
'''

def levenshtein_ids(A, B, insertion=1, deletion=1, substitution=1):
    """
    Compute number of insertions deletions, and substitutions for an
    optimal alignment.
    There may be more than one, in which case we disfavor substitution.
    >>> print levenshtein_ids('sitting', 'kitten')
    (1, 0, 2)
    >>> print levenshtein_ids('banat', 'banap')
    (0, 0, 1)
    >>> print levenshtein_ids('Saturday', 'Sunday')
    (2, 0, 1)
    """
    # basic checks
    if len(A) == len(B) and A == B:
        return (0, 0, 0)
    if len(B) > len(A):
        (A, B) = (B, A)
    if len(A) == 0:
        return len(B)


'''

def hist_sub_del_ins():
    df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/split/all/all_test.csv', sep=',')
    df = df.astype(str)
    df = df.replace(['é', 'è', 'È', 'É'], 'e', regex=True)
    df = df.replace(['à', 'â', 'Â'], 'a', regex=True)
    substitution=0
    insertion=0
    deletion=0
    correct=0

    for i in range(len(df)):
        if df.manual_raw_value[i] != df.raw_value[i]:
            text = df.manual_raw_value[i]
            text2 = df.raw_value[i]
            x,y,z= levenshtein_ids(text,text2)
            insertion +=x
            deletion +=y
            substitution +=z

        else:
            correct +=1
    x = (1, 2, 3)
    y = (substitution, insertion, deletion)
    z=substitution+insertion+deletion
    num_char=pd.Series(list(df.manual_raw_value.str.cat())).count()

    acc=100*(1-(z/num_char))
    labels= ["Substitution", "Insertion","Deletion"]

    int_labels=[1,2,3]
    plt.title("Type of errors")
    plt.bar(int_labels,y, align='center')
    plt.xticks(int_labels,labels)
    plt.show()
    ''''
    plt.xlabel('type of edit')
    plt.ylabel('Frequency')
    for i in range(len(y)):
        plt.vlines(y[i], 0, x[i])  # Here you are drawing the horizontal lines
    plt.show()
    '''

    size = [substitution, insertion, deletion]
    colors = ['yellowgreen', 'gold', 'lightskyblue']
    explode = (0.1, 0, 0)
    # patches, texts = plt.pie(size, colors=colors, shadow=True, startangle=90)
    plt.pie(size, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    # plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    # plt.tight_layout()
    plt.show()
    return substitution, insertion, deletion, num_char,acc




