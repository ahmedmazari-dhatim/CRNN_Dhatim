import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob
import pandas as pd

path_train='/home/ahmed/Pictures/cogedis/24072017/all/'

path_valid='/home/ahmed/Pictures/cogedis/24072017/all/'

path_output_train = '/home/ahmed/Pictures/cogedis/24072017/maj+min/train/'
path_output_valid=  '/home/ahmed/Pictures/cogedis/24072017/maj+min/valid/'



df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/maj+min/all_maj_min_train.csv',sep=',')
df = df.astype(str)
os.chdir(path_train)

images_name = glob.glob("*.png")
set_img = set([x.rsplit('.', 1)[0] for x in images_name])
labels_train=[]
images_train=[]
for img in set_img:
    if img in df.id.values:
    #if (img in df.id.values) and (img+'.png' in images_name):    # in case of data augmentation
        label = df.loc[df.id == img, 'manual_raw_value'].item()
        labels_train.append(label)
        image = img + '.png'
        images_train.append(image)
train_img=images_train
train_label=labels_train


df_valid=pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/maj+min/all_maj_min_valid.csv',sep=',')
df_valid=df_valid.astype(str)

os.chdir(path_valid)
images_name_valid = glob.glob("*.png")
set_img_valid = set([x.rsplit('.', 1)[0] for x in images_name_valid])
labels_valid=[]
images_valid=[]
for img in set_img_valid:
    if img in df_valid.id.values:
        label = df_valid.loc[df_valid.id == img, 'manual_raw_value'].item()
        labels_valid.append(label)
        image = img + '.png'
        images_valid.append(image)
valid_img=images_valid
valid_label=labels_valid


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)



def createDataset(outputPath, images_train, labels_train,path, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(images_train) == len(labels_train))
    nSamples = len(images_train)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = images_train[i]
        label = labels_train[i]
        if not os.path.exists(path + imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(path + imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
   createDataset(path_output_train, train_img, train_label,path_train)
   createDataset(path_output_valid, valid_img, valid_label,path_valid)



