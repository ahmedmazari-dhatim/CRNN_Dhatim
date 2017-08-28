

from __future__ import division
import cv2
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import torch.nn as nn
import models.crnn as crnn
#import timey
from datetime import datetime
import glob
import os
import pandas as pd



df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/split/digit+alphabet/digit+alphabet_test.csv',sep=",")

#df['crnn']=None
df = df.astype(str)
#test_path='/home/ahmed/Downloads/crnn.pytorch-master/data/IIITS5K/'

test_path='/home/ahmed/Pictures/cogedis/24072017/all/'
#model_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/crnn.pth'
#model_path='/home/ahmed/Pictures/model/augmented_alpha_digit/netCRNN_64_65.pth'
model_path='/home/ahmed/Pictures/cogedis/24072017/split/digit+alphabet/model/output_transfer_learning/output_transfer_learning/netCRNN_9_353.pth'

#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/demo.png'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/logo.png'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/ShareImg.png'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/char.png'
#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/awe.jpg'

#digits

#img_path = '/home/ahmed/Downloads/crnn.pytorch-master/data/digits3.png'

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
#t = datetime.now()
p=0
n=0
#crnn_predicted=[]
df['crnn_pred']=None
H=0


for img in images_names:
    if img.rsplit('.',1)[0] in df.id.values:
        image = Image.open(img).convert('L')
        id, ext = img.rsplit('.', 1)
        # start = time.time()
        a = datetime.now()

        image = transformer(image).cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        m=torch.nn.Softmax()

        model.eval()
        preds = model(image)
        temps=preds.cpu()
#        prob=torch.max(m(temps)*100)
#        arr = temps.data.numpy()

 #       print("array ", arr)
        '''
        arr = preds.data.numpy()
        for i in range(0, len(temps)):
            if arr[i] != 0:
                prob = torch.max(m(temps[i]) * 100)
                print("proba ", prob)
        '''
        #cuda_tensor = torch.rand(5).cuda()
        #np_array = cuda_tensor.cpu().numpy()
        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        idx = df[df['id'] == id].index.values
        df.loc[df.index[idx], 'crnn_pred'] = sim_pred
        # df.iloc[idx, df.columns.get_loc(4)] = sim_pred
        # crnn_predicted.append(sim_pred)
        # end = time.time()
        b = datetime.now()
        c = b - a
        print('%-20s => %-20s' % (raw_pred, sim_pred), img)
        if sim_pred == df.loc[df.id == id, 'manual_raw_value'].item():
            H += 1

        # print(end - start)
        print(c.total_seconds(), " seconds")
        print(c.total_seconds() * 1000, " miliseconds")

        '''
        if (sim_pred+'.png') == img:
            p +=1
        '''
        # print(c)
        '''
        arr=preds.data.numpy()
        for i in range (0,len(temps)):
            if arr[i] != 0:
                prob=torch.max(m(temps[i])*100)
                print("proba ", prob)
        '''
        i += 1

n += c.total_seconds()
#df.to_csv('/home/ahmed/Pictures/cogedis/data_crnn/augmented_proba_ctc_alpha_digit.csv',sep=',')
print("total images ", i)
print("correct pre ",H)
print ("accuracy", (H/i)*100)
df.to_csv('/home/ahmed/Pictures/cogedis/24072017/split/digit+alphabet/digit+alphabet_test_crnn_pred.csv',sep=',',index=False)
print(n,"estimated time for prediciton")

