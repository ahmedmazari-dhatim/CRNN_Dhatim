# coding: utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
from datetime import datetime
import models.crnn as crnn
from logger import Logger
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
a = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=65, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1, help='learning rate for Critic, default=0.05')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=1, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')

parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--save_folder', default='/home/ahmed/crnn/log-model/', help='Location to save epoch models')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log_dir', default='/home/ahmed/crnn/visualize', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
opt = parser.parse_args()
save_folder = opt.save_folder
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
loss_results, cer_results, wer_results = torch.Tensor(opt.niter), torch.Tensor(opt.niter), torch.Tensor(
opt.niter)
'''
if opt.visdom:
    from visdom import Visdom
    vis = Visdom()
    opts = [dict(title='Loss', ylabel='Loss', xlabel='Epoch'),
            dict(title='WER', ylabel='WER', xlabel='Epoch'),
            dict(title='CER', ylabel='CER', xlabel='Epoch')]
    viz_windows = [None, None, None]
    epochs = torch.arange(1,opt.niter + 1)

if opt.tensorboard:
    from logger import TensorBoardLogger
    try:
        os.mkdir(opt.log_dir)
    except OSError as e:
        if e.errno == e.errno.EEXIST:
            print('Directory already exists.')
            for file in os.listdir(opt.log_dir):
                file_path = os.path.join(opt.log_dir, file)
                try :
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    raise
        else :
            raise
    logger = TensorBoardLogger(opt.log_dir)
try:
    os.mkdir(save_folder)
except OSError as e:
    if e.errno == e.errno.EEXIST:
        print('Directory already exists.')
    else:
        raise





if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# Set the logger

'''

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
    collate_fn=dataset.alignCollate(imgH=opt.imgH, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

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
    crnn.load_state_dict(torch.load(opt.crnn))
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


def val(net, dataset, criterion, max_iter=500000):
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

    #for i in range(max_iter):
    loss_dis =[]
    for i in range(min(max_iter, len(data_loader))):
        #print(i)
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
        loss_dis.append(cost)

        _, preds = preds.max(2)
        print("here is preds max ", preds)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)

    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    print('number of correct ', n_correct)
    plt.figure()
    plt.plot(loss_dis)
    # ====================TensorBoard logging =============#
    # (1) Log the scalar values



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


for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:

            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
    b = datetime.now()
    c=b-a
    print(c,"estimated time")
    print(c.total_seconds(),"in second")
