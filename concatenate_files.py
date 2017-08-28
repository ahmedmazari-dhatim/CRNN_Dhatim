import os
import glob
import pandas as pd

def concatenate(indir,outputfile):
    os.chdir(indir)
    fileList=glob.glob("*.csv")
    dfList=[]
    i=0
    #colnames=["id","ocr","raw_value","manual_raw_value"]
    for filename in fileList:
        #print(filename)
        #df=pandas.read_csv(input,header=None)
        df = pd.read_csv(filename, sep=",",index_col=None, error_bad_lines=False)

        df[~df.isnull()]
        df.dropna()
        #df=pd.read_csv(filename,index_col=None,sep=";")
        #df = df.astype(str)
        i += 1
        print(filename, " is ", i)
        print('size', len(df))
        dfList.append(df)
        print("great ", i)
    #concatenateDF=pandas.concat(dfList,axis=0)
    #concatenateDF.columns=colnames
    #concatenateDF.to_save(outputfile,index=None)

    df = pd.concat(dfList, axis=0)
    df.to_csv(output,sep=",",index=None)

    #df.columns = ["id", "ocr", "raw_value", "manual_raw_value"]

    #print(df.head())
    #print("ok ")

    df = pd.read_csv(output, sep=",", index_col=None,
                     error_bad_lines=False)
    df.to_csv(output,sep=",",index=None)

if __name__ == "__main__":
    #input="/home/ahmed/Pictures/cogedis/ne"
    #input="/home/ahmed/Pictures/cogedis/24072017/"
    #output="/home/ahmed/Pictures/cogedis/24072017/concatenated.csv"
    input="/home/ahmed/Pictures/cogedis/words/"
    output="/home/ahmed/Pictures/cogedis/words/concatenated.csv"
    concatenate(input,output)
    #len(df[pd.isnull(df.manual_raw_value)])
    #df.dropna(how='any')
