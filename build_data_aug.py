import pandas as pd
#tech = ['sharpen','elasticdeformation','horizontalline','diagonalline','diagonalinverseline','verticalleftline','verticalrightline','severalrowsline',
 #    'severalcolsline', 'severalcolsrowsline', 'superpixel', 'gaussianblur', 'additivegaussiannoise','dropout','translation','rotation90','rotation-90','rotation5','rotation-5']

tech2 = ['horizontalline','verticalleftline','verticalrightline','translation','superpixel', 'additivegaussiannoise',
        'elasticdeformation','gaussianblur','sharpen','rotation5','rotation-5','dropout']


#df = pd.read_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/words.csv')

#df = pd.read_csv('/home/ahmed/Pictures/cogedis/data_crnn/augmented_without_test/digit.csv')

df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/split/all/all_valid.csv')
df = df.astype(str)
df=df.ix[:,0:4]
df3=df[~df.manual_raw_value.str.match(r'^[\d,:.+\'%/-]*$')]
df = df[df.manual_raw_value.str.match(r'^[\d,:.+\'%/-]*$')]
#df = pd.read_csv('/home/ahmed/Pictures/cogedis/data_crnn/augmented_without_test/digit.csv')
dfs = []
def f(x):
    df = pd.DataFrame({'id':[x['id'] + '_' + t for t in tech2],
                       'ocr':x['ocr'],
                       'manual_raw_value':x['manual_raw_value'],
                       'raw_value':x['raw_value']})
    #print (df)
    dfs.append(df)

df.apply(f, axis=1)

df1 = pd.concat(dfs)
#print (df1)

df2 = pd.concat([df, df1,df3], ignore_index=True).reindex_axis(df.columns, axis=1)
#df2.to_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/words_augmented.csv',sep=',')
df2.to_csv('/home/ahmed/Pictures/cogedis/24072017/split/all/all_valid_augmented.csv',sep=',',index=False)

