from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import chardet
import keys
import sys
import collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
reload(sys)
sys.setdefaultencoding('utf8')

from datetime import datetime
import models.crnn as crnn
a = datetime.now()

str1=keys.alphabet
parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')

parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')

parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=13, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1.0, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default=str1)
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=1, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true',default=True, help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

#opt.manualSeed = random.randint(1, 10000)  # fix seed
opt.manualSeed=100
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

opt.saveInterval= len(train_loader)
ngpu = int(opt.ngpu)
nh = int(opt.nh)
alphabet = opt.alphabet
nclass = len(alphabet) + 1
nc = 1

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

crnn = crnn.CRNN(opt.imgH, nc, nclass, nh, ngpu)
crnn.apply(weights_init)

if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    pre_trainmodel = torch.load(opt.crnn)
    model_dict = crnn.state_dict()
    for k,v in model_dict.items():
        if not (k == 'rnn.1.embedding.weight' or k == 'rnn.1.embedding.bias'):
            model_dict[k] = pre_trainmodel[k]
    crnn.load_state_dict(model_dict)
print(crnn)


image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
    
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
       # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

  
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    print('number of correct ', n_correct)
    
    return loss_avg.val(),accuracy


#  val(crnn, test_dataset, criterion)
#  exit(0)

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()

    return cost 



g_i= []
g_train_loss= []
g_valid_loss=[]
g_valid_accuracy=[]
g_iter=[]
g_iter_valid=[]
g_iter_train=[]
g_iter_last_epoch=[]
g_iter_train_last_epoch=[]
g_iter_valid_last_epoch=[]
g_iter_valid_accuracy_first_epoch=[]
g_iter_valid_accuracy_last_epoch=[]


for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        x=loss_avg.val()

        if epoch==0:

            g_iter=np.append(g_iter,i)
            cost_val, accuracy_valid = val(crnn, test_dataset, criterion)
            g_iter_train=np.append(g_iter_train,x)
            g_iter_valid=np.append(g_iter_valid,cost_val)
            g_iter_valid_accuracy_first_epoch=np.append(g_iter_valid_accuracy_first_epoch,accuracy_valid)
        if epoch==(opt.niter-1):

            g_iter_last_epoch=np.append(g_iter_last_epoch,i)
            cost_val, accuracy_valid = val(crnn, test_dataset, criterion)
            g_iter_train_last_epoch=np.append(g_iter_train_last_epoch,x)
            g_iter_valid_last_epoch=np.append(g_iter_valid_last_epoch,cost_val)
            g_iter_valid_accuracy_last_epoch=np.append(g_iter_valid_accuracy_last_epoch,accuracy_valid)



        i += 1


        if i % opt.displayInterval == 0:

            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()


        if i % opt.valInterval == 0:
            cost_val,accuracy_valid=val(crnn, test_dataset, criterion)
      

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
        if (epoch==opt.niter):
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}_ok.pth'.format(opt.experiment, epoch, i))
    g_i = np.append(g_i, epoch)
    g_train_loss = np.append(g_train_loss,x)

    g_valid_loss = np.append(g_valid_loss, cost_val)

    g_valid_accuracy = np.append(g_valid_accuracy, accuracy_valid)              
    b = datetime.now()
    c=b-a
    print(c,"estimated time")
    print(c.total_seconds(),"in second")


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4,ax4  = plt.subplots()
fig5,ax5  = plt.subplots()
fig6,ax6  = plt.subplots()
fig7,ax7  = plt.subplots()

ax1.plot(g_i,g_train_loss,label='train_loss')
ax1.plot(g_i,g_valid_loss,label='valid_loss')
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.set_title(" loss on CRNN  traind with abby digit ")
ax1.legend(loc='best')

fig1.savefig(opt.experiment+'/try_loss.png', bbox_inches='tight')


ax2.plot(g_i,g_valid_accuracy,label='valid_acc')
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.set_title(" loss on CRNN ")
ax2.legend(loc='best')
fig2.savefig(opt.experiment+'/try_acc.png', bbox_inches='tight')

ax3.plot(g_i,g_valid_loss,label='valid_acc')
ax3.set_xlabel("epoch")
ax3.set_ylabel("loss")
ax3.set_title("CRNN+ validation loss")
ax3.legend(loc='best')
fig3.savefig(opt.experiment+'/valid_loss.png', bbox_inches='tight')

print('g_iter : ',g_iter)

ax4.plot(g_iter,g_iter_train,label='train_loss')
ax4.plot(g_iter,g_iter_valid,label='valid_loss')
ax4.set_xlabel("iterations")
ax4.set_ylabel("loss")
ax4.set_title(" train and valid loss on first  epoch ")
ax4.legend(loc='best')
fig4.savefig(opt.experiment+'/train_valid_loss_first_epoch.png', bbox_inches='tight')

ax5.plot(g_iter_last_epoch,g_iter_train_last_epoch,label='train_loss')
ax5.plot(g_iter_last_epoch,g_iter_valid_last_epoch,label='valid_loss')
ax5.set_xlabel("iterations")
ax5.set_ylabel("loss")
ax5.set_title(" loss on train and validation last epoch ")
ax5.legend(loc='best')
fig5.savefig(opt.experiment+'/train_valid_loss_last_epoch.png', bbox_inches='tight')


ax6.plot(g_iter_last_epoch,g_iter_valid,label='valid_loss_first_epoch')
ax6.plot(g_iter_last_epoch,g_iter_valid_last_epoch,label='valid_loss_last_epoch')
ax6.plot(g_iter_last_epoch,g_iter_train,label='train_loss_first_epoch')
ax6.plot(g_iter_last_epoch,g_iter_train_last_epoch,label='train_loss_last_epoch')
ax6.set_xlabel(" iterations")
ax6.set_ylabel("loss")
ax6.set_title(" loss on first and last epoch : train and validation ")
ax6.legend(loc='best')
fig6.savefig(opt.experiment+'/train_valid_loss_last_first_epoch.png', bbox_inches='tight')

ax7.plot(g_iter_last_epoch,g_iter_valid_accuracy_first_epoch,label='valid_accuracy_first_epoch')
ax7.plot(g_iter_last_epoch,g_iter_valid_accuracy_last_epoch,label='valid_accuracy_last_epoch')
ax7.set_xlabel(" iteration")
ax7.set_ylabel("accuracy")
ax7.set_title(" accuracy on first and last epoch")
ax7.legend(loc='best')
fig7.savefig(opt.experiment+'/valid_accuracy_last_first_epoch.png', bbox_inches='tight')

