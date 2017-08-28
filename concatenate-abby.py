import os
import glob
import pandas

def concatenate(indir,outfile):
    os.chdir(indir)
    fileList=glob.glob("words_*.csv")
    dfList=[]
    colnames=["id","ocr","raw_value","manual_raw_value"]
    for filaname in fileList:
        print(filaname)
        df=pandas.read_csv(filaname,header=None)
        df=df.astype(str)
        print(df.shape)
        dfList.append(df)

    concatDF=pandas.concat(dfList,axis=0)
    concatDF.columns=colnames
    concatDF.to_csv(outfile,index=None)

def abby_data(indir):
    df = pandas.read_csv(indir, sep=',')
    df = df.astype(str)
    df=df.replace(['é','è','È','É'],'e', regex=True)
    df = df.replace(['à','â','Â'], 'a', regex=True)
    df=df.loc[df.ocr=='ABBYY']
    a = r'^[\d]+$'
    df_digit = df[df.manual_raw_value.str.match(a)]
    df_digit.to_csv("/home/ahmed/Pictures/cogedis/words/ABBY_DIGIT.csv", index=False, sep=',')
    df_dig_alpha = df[df.manual_raw_value.str.match(r'^[\da-zA-Z]*$')]
    df_dig_alpha.to_csv("/home/ahmed/Pictures/cogedis/words/ABBY_DIGIT_ALPAHBET.csv", index=False, sep=',')


if __name__ == "__main__":
    input="/home/ahmed/Pictures/cogedis/words/"
    output="/home/ahmed/Pictures/cogedis/words/concatenated.csv"

    concatenate(input,output)
    abby_data(output,)
'''
#process words
df=pandas.read_csv("/home/ahmed/Pictures/cogedis/words/words.csv",sep=",")
df=df.astype(str)
df.columns = range(0, df.shape[1])
for i in range(4, df.shape[1]):
    df = df[df.iloc[:, i].isnull()]
#df = df[[0, 1, 2, 3]]
df.to_csv("/home/ahmed/Pictures/cogedis/words/words_processed.csv",sep=",",index=None,header=None)

#process words1

df2=pandas.read_csv("/home/ahmed/Pictures/cogedis/words/words1.csv",sep=",")
df2=df2.astype(str)
df2.columns = range(0, df2.shape[1])
for i in range(5, df2.shape[1]):
    df2 = df2[df2.iloc[:, i].isnull()]
df2 = df2[[0, 1, 2, 3]]
df2.to_csv("/home/ahmed/Pictures/cogedis/words/words_processed1.csv",sep=",",index=None,header=None)
#df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
'''