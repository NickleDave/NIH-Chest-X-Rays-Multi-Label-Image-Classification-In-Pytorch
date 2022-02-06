import argparse
import os, sys, time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models 

from datasets import XRaysTrainDataset  
from datasets import XRaysTestDataset
from trainer import fit, eval_
import config

def q(text = ''): # easy way to exiting the script. useful while debugging
    print('> ', text)
    sys.exit()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'\ndevice: {device}')
    
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default = 'NIH Chest X-rays', help = 'This is the path of the training data')
parser.add_argument('--model', type=str, default='resnet50', choices={'alexnet', 'vgg16', 'resnet50'})
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--bs', type = int, default = 128, help = 'batch size')
parser.add_argument('--lr', type = float, default = 1e-5, help = 'Learning Rate for the optimizer')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--stage', type = int, default = 1, help = 'Stage, it decides which layers of the Neural Net to train')
parser.add_argument('--epochs', type = int, default=20)
parser.add_argument('--loss_func', type = str, default = 'FocalLoss', choices = {'BCE', 'FocalLoss'}, help = 'loss function')
parser.add_argument('-r','--resume', action = 'store_true') # args.resume will return True if -r or --resume is used in the terminal
parser.add_argument('--ckpt', type = str, help = 'Path of the ckeckpoint that you wnat to load')
parser.add_argument('-t','--test', action = 'store_true')   # args.test   will return True if -t or --test   is used in the terminal
args = parser.parse_args()

if args.resume and args.test:
    q('The flow of this code has been designed either to train the model or to test it.\nPlease choose either --resume or --test')

stage = args.stage
if not args.resume:
    print(f'\nOverwriting stage to 1, as the model training is being done from scratch')
    stage = 1
    
if args.test:
    print('TESTING THE MODEL')
else:
    if args.resume:
        print('RESUMING THE MODEL TRAINING')
    else:
        print('TRAINING THE MODEL FROM SCRATCH')

script_start_time = time.time() # tells the total run time of this script

# mention the path of the data
data_dir = os.path.join('data',args.data_path) # Data_Entry_2017.csv should be present in the mentioned path

# define a function to count the total number of trainable parameters
def count_parameters(model): 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

# make the datasets
XRayTrain_dataset = XRaysTrainDataset(data_dir, transform = config.transform)
train_percentage = 0.8
train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])

XRayTest_dataset = XRaysTestDataset(data_dir, transform = config.transform)

print('\n-----Initial Dataset Information-----')
print('num images in train_dataset   : {}'.format(len(train_dataset)))
print('num images in val_dataset     : {}'.format(len(val_dataset)))
print('num images in XRayTest_dataset: {}'.format(len(XRayTest_dataset)))
print('-------------------------------------')

# make the dataloaders
batch_size = args.bs # 128 by default
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,
                                           shuffle = True, num_workers=args.num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size,
                                         shuffle = False, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size = batch_size,
                                          shuffle = False,  num_workers=args.num_workers)

print('\n-----Initial Batchloaders Information -----')
print('num batches in train_loader: {}'.format(len(train_loader)))
print('num batches in val_loader  : {}'.format(len(val_loader)))
print('num batches in test_loader : {}'.format(len(test_loader)))
print('-------------------------------------------')

# sanity check
if len(XRayTrain_dataset.all_classes) != 15: # 15 is the unique number of diseases in this dataset
    q('\nnumber of classes not equal to 15 !')

a,b = train_dataset[0]
print('\nwe are working with \nImages shape: {} and \nTarget shape: {}'.format( a.shape, b.shape))

# make models directory, where the models and the loss plots will be saved
if not os.path.exists(config.models_dir):
    os.mkdir(config.models_dir)

# define the loss function
if args.loss_func == 'FocalLoss': # by default
    from losses import FocalLoss
    loss_fn = FocalLoss(device = device, gamma = 2.).to(device)
elif args.loss_func == 'BCE':
    loss_fn = nn.BCEWithLogitsLoss().to(device)
else:
    raise ValueError(
        f'invalid loss function: {args.loss_func}'
    )

# define the learning rate
lr = args.lr

if not args.test: # training

    # initialize the model if not args.resume
    if not args.resume:
        print('\nnot loading checkpoint')
        # import pretrained model
        print(f'Loading model: {args.model}. Weights pre-trained on ImageNet: {args.pretrained}')
        if args.model == 'resnet50':
            model = models.resnet50(pretrained=args.pretrained)
        elif args.model == 'alexnet':
            model = models.alexnet(pretrained=args.pretrained)
        elif args.model == 'vgg16':
            model = models.vgg16(pretrained=args.pretrained)
        else:
            raise ValueError(
                f'invalid model name: {args.model}'
            )

        # change the last linear layer
        num_classes = len(XRayTrain_dataset.all_classes)
        if args.model == 'resnet50':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model.to(device)

        if args.pretrained:  # only training 'layer2', 'layer3', 'layer4' and 'fc'
            if args.model == 'resnet50':
                print('freezing some pretrained weights of resnet50 model')
                for name, param in model.named_parameters(): # all requires_grad by default, are True initially
                    # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                    if ('layer2' in name) or ('layer3' in name) or ('layer4' in name) or ('fc' in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        # since we are not resuming the training of the model
        epochs_till_now = 0

        # making empty lists to collect all the losses
        losses_dict = {
            'epoch_train_loss': [],
            'epoch_val_loss': [],
            'total_train_loss_list': [],
            'total_val_loss_list': []
        }

    else:
        if args.ckpt == None:
            q('ERROR: Please select a valid checkpoint to resume from')
            
        print('\nckpt loaded: {}'.format(args.ckpt))
        ckpt = torch.load(os.path.join(config.models_dir, args.ckpt)) 

        # since we are resuming the training of the model
        epochs_till_now = ckpt['epochs']
        model = ckpt['model']
        model.to(device)
        
        # loading previous loss lists to collect future losses
        losses_dict = ckpt['losses_dict']

    # printing some hyperparameters
    print('\n> loss_fn: {}'.format(loss_fn))
    print('> epochs_till_now: {}'.format(epochs_till_now))
    print('> batch_size: {}'.format(batch_size))
    print('> stage: {}'.format(stage))
    print('> lr: {}'.format(lr))

else: # testing
    if args.ckpt == None:
        q('ERROR: Please select a checkpoint to load the testing model from')
        
    print('\ncheckpoint loaded: {}'.format(args.ckpt))
    ckpt = torch.load(os.path.join(config.models_dir, args.ckpt)) 

    # since we are resuming the training of the model
    epochs_till_now = ckpt['epochs']
    model = ckpt['model']
    
    # loading previous loss lists to collect future losses
    losses_dict = ckpt['losses_dict']

    eval_(device, test_loader, model, loss_fn, log_interval = 1)

# make changes(freezing/unfreezing the model's layers) in the following, for training the model for different stages 
if not args.test and args.resume:

    if stage == 1:

        print('\n----- STAGE 1 -----') # only training 'layer2', 'layer3', 'layer4' and 'fc'
        if args.model == 'resnet50':
            for name, param in model.named_parameters(): # all requires_grad by default, are True initially
                # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                if ('layer2' in name) or ('layer3' in name) or ('layer4' in name) or ('fc' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print(f'no stages for model: {args.model}')

    elif stage == 2:

        print('\n----- STAGE 2 -----') # only training 'layer3', 'layer4' and 'fc'

        if args.model == 'resnet50':
            for name, param in model.named_parameters():
                # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                if ('layer3' in name) or ('layer4' in name) or ('fc' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print(f'no stages for model: {args.model}')

    elif stage == 3:

        print('\n----- STAGE 3 -----') # only training  'layer4' and 'fc'
        if args.model == 'resnet50':
            for name, param in model.named_parameters():
                # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                if ('layer4' in name) or ('fc' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print(f'no stages for model: {args.model}')

    elif stage == 4:

        print('\n----- STAGE 4 -----') # only training 'fc'
        if args.model == 'resnet50':
            for name, param in model.named_parameters():
                # print('{}: {}'.format(name, param.requires_grad)) # this shows True for all the parameters
                if ('fc' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print(f'no stages for model: {args.model}')

if not args.test:
    # checking the layers which are going to be trained (irrespective of args.resume)
    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            layer_name = str.split(name, '.')[0]
            if layer_name not in trainable_layers: 
                trainable_layers.append(layer_name)
    print('\nfollowing are the trainable layers...')
    print(trainable_layers)

    print('\nwe have {} Million trainable parameters here in the {} model'.format(
        count_parameters(model),
        model.__class__.__name__)
    )

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

# make changes in the parameters of the following 'fit' function
fit(device, XRayTrain_dataset, train_loader, val_loader,    
                                        test_loader, model, loss_fn, 
                                        optimizer, losses_dict,
                                        epochs = args.epochs,
                                        log_interval = 25, save_interval = 1)

script_time = time.time() - script_start_time
m, s = divmod(script_time, 60)
h, m = divmod(m, 60)
print('script run time: {} h {}m'.format(int(h), int(m)))

# ''' 
# This is how the model is trained...
# ##### STAGE 1 ##### FocalLoss lr = 1e-5
# training layers = layer2, layer3, layer4, fc 
# epochs = 2
# ##### STAGE 2 ##### FocalLoss lr = 3e-4
# training layers = layer3, layer4, fc 
# epochs = 5
# ##### STAGE 3 ##### FocalLoss lr = 7e-4
# training layers = layer4, fc 
# epochs = 4
# ##### STAGE 4 ##### FocalLoss lr = 1e-3
# training layers = fc 
# epochs = 3
# '''
