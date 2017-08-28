# !/usr/bin/python2.7
# -*-coding:latin-1*
import pandas as pd




def  group_data():
    def find_group(val):
        val = str(val)
        val = val.lower()

        if val.isalpha():
            return 'Alpha'
        elif val.isdigit():
            return 'digit'
        elif val.isalnum and any(c.isalpha() for c in val) and any(c.isdigit() for c in val):

            return 'Alphanumeric'
        else:
            return 'Special'


    df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/split/all/OMNIPAGE.csv', sep=',')
    df = df.astype(str)
    df = df.replace(['é', 'è', 'È', 'É'], 'e', regex=True)
    df = df.replace(['à', 'â', 'À'], 'a', regex=True)
    df.manual_raw_value = df.manual_raw_value.str.lower()

    df.manual_raw_value.apply(find_group)

    df.manual_raw_value.apply(find_group).value_counts().plot(kind='bar')

def class_data():
    #df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/split/alphabet/alphabet.csv', sep=',')
    df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/split/all/OMNIPAGE.csv', sep=',')
    df = df.astype(str)
    df = df.replace(['é', 'è', 'È', 'É'], 'e', regex=True)
    df = df.replace(['à', 'â', 'À'], 'a', regex=True)
    df.manual_raw_value = df.manual_raw_value.str.lower()

    classes=set(df.manual_raw_value.apply(list).sum())
    print("number of classes is ", len(classes))
    #print("classes are " ,classes)
    print("classes are ", [x.decode("utf-8") for x in classes] )

    pd.Series(list(df.manual_raw_value.str.cat())).value_counts().plot(kind="bar")


if __name__ == "__main__":
    group_data()
    #class_data()

