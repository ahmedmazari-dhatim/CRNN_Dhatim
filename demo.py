

from __future__ import division
import cv2
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import torch.nn as nn
import models.crnn as crnn

from datetime import datetime
import glob
import os
import pandas as pd



df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/split/all/all_processed_clean.csv',sep=",")


df = df.astype(str)


test_path='/home/ahmed/Pictures/cogedis/24072017/all/'

model_path='/home/ahmed/Pictures/cogedis/24072017/split/digit/ocr/ab+omni/model/output_transfer_final/netCRNN_14_71.pth'



#alphabet = 'abcdefghijklmnopqrstuvwxyz'
#alphabet = "0123456789,:%'./+-"
#alphabet = '0123456789'
#alphabet="0123456789,:.+%"
#alphabet="abcdefghijklmnopqrstuvwxyz0123456789.,:+'/%-" #all
alphabet="abcdefghijklmnopqrstuvwxyz0123456789" # digit + alphabet
#alphabet="abcdefghijklmnopqrstuvwxyz.,:+'/%-" #alphabet+char
model = crnn.CRNN(32, 1, len(alphabet)+1,256, 1).cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))

os.chdir(test_path)
images_names=glob.glob("*.png")
i=0

p=0
n=0

df['crnn_pred']=None
H=0


for img in images_names:
    if img.rsplit('.',1)[0] in df.id.values:
        image = Image.open(img).convert('L')
        id, ext = img.rsplit('.', 1)
        
        a = datetime.now()

        image = transformer(image).cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        m=torch.nn.Softmax()

        model.eval()
        preds = model(image)
        '''
        def softmax(z):
            return np.exp(z-np.max(z,0))/np.sum(np.exp(z-np.max(z,0)),0.)
        #Transform logits to probabilities
        # we first define softmax function 
        preds=preds.cpu().data.numpy()
        preds=preds.squeeze()
        preds_proba=softmax(preds)
        '''
        temps=preds.cpu()
        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        idx = df[df['id'] == id].index.values
        df.loc[df.index[idx], 'crnn_pred'] = sim_pred

        b = datetime.now()
        c = b - a
        print('%-20s => %-20s' % (raw_pred, sim_pred), img)
        if sim_pred == df.loc[df.id == id, 'manual_raw_value'].item():
            H += 1

        
        print(c.total_seconds(), " seconds")
        print(c.total_seconds() * 1000, " miliseconds")

 
        i += 1

n += c.total_seconds()
print("total images ", i)
print("correct pre ",H)
print ("accuracy", (H/i)*100)
df.to_csv('/home/ahmed/Pictures/cogedis/24072017/split/digit/ocr/ab+omni/digit_test_crnn_pred.csv',sep=',',index=False)
print(n,"estimated time for prediciton")

