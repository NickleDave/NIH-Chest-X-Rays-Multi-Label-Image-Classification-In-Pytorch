import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os, time
import numpy as np
import torch
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import config


def get_roc_auc_score(y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
        all_classes = pickle.load(handle)
    
    NoFindingIndex = all_classes.index('No Finding')

    print('\nNoFindingIndex: ', NoFindingIndex)
    print('y_true.shape, y_probs.shape ', y_true.shape, y_probs.shape)
    GT_and_probs = {'y_true': y_true, 'y_probs': y_probs}

    class_roc_auc_list = []    
    useful_classes_roc_auc_list = []
    
    for i in range(y_true.shape[1]):
        class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        class_roc_auc_list.append(class_roc_auc)
        if i != NoFindingIndex:
            useful_classes_roc_auc_list.append(class_roc_auc)
    if True:
        print('\nclass_roc_auc_list: ', class_roc_auc_list)
        print('\nuseful_classes_roc_auc_list', useful_classes_roc_auc_list)

    roc_auc = np.mean(np.array(useful_classes_roc_auc_list))
    return GT_and_probs, roc_auc


def make_plot(epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list):
    '''
    This function makes the following 4 different plots-
    1. mean train loss VS number of epochs
    2. mean val   loss VS number of epochs
    3. batch train loss for all the training   batches VS number of batches
    4. batch val   loss for all the validation batches VS number of batches
    '''
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('epoch train loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('epoch train loss')
    ax1.plot(epoch_train_loss)

    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(epoch_val_loss)

    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(total_train_loss_list)

    ax4.title.set_text('batch val loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch val loss')
    ax4.plot(total_val_loss_list)
    
    return fig

    
def train_epoch(device, train_loader, model, loss_fn, optimizer, epochs_till_now, final_epoch, log_interval):
    '''
    Takes in the data from the 'train_loader', calculates the loss over it using the 'loss_fn' 
    and optimizes the 'model' using the 'optimizer'  
    
    Also prints the loss and the ROC AUC score for the batches, after every 'log_interval' batches. 
    '''
    model.train()

    running_train_loss = 0
    train_loss_list = []

    start_time = time.time()
    pbar = tqdm(train_loader)
    for batch_idx, (img, target) in enumerate(pbar):

        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()    
        out = model(img)        
        loss = loss_fn(out, target)
        running_train_loss += loss.item()*img.shape[0]
        train_loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        batch_time = time.time() - start_time
        m, s = divmod(batch_time, 60)
        pbar.set_description(
            'Train Loss for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(
                str(batch_idx+1).zfill(3),
                str(len(train_loader)).zfill(3),
                epochs_till_now,
                final_epoch,
                round(loss.item(), 5),
                int(m),
                round(s, 2))
        )
        
        start_time = time.time()
            
    return train_loss_list, running_train_loss / float(len(train_loader.dataset))

def val_epoch(device, val_loader, model, loss_fn, epochs_till_now = None, final_epoch = None, log_interval = 1):
    '''
    It essentially takes in the val_loader/test_loader, the model and the loss function and evaluates
    the loss and the ROC AUC score for all the data in the dataloader.
    
    It also prints the loss and the ROC AUC score for every 'log_interval'th batch, only when 'test_only' is False
    '''
    model.eval()

    running_val_loss = 0
    val_loss_list = []
    val_loader_examples_num = len(val_loader.dataset)

    probs = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
    gt = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
    k = 0

    with torch.no_grad():
        batch_start_time = time.time()
        pbar = tqdm(val_loader)
        for batch_idx, (img, target) in enumerate(pbar):
            img = img.to(device)
            target = target.to(device)    

            out = model(img)        
            loss = loss_fn(out, target)    
            running_val_loss += loss.item()*img.shape[0]
            val_loss_list.append(loss.item())

            # storing model predictions for metric evaluation
            probs[k: k + out.shape[0], :] = out.cpu()
            gt[k: k + out.shape[0], :] = target.cpu()
            k += out.shape[0]

            if (batch_idx + 1) % log_interval == 0:
                batch_time = time.time() - batch_start_time
                m, s = divmod(batch_time, 60)
                pbar.set_description(
                    'Val Loss for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(
                        str(batch_idx+1).zfill(3),
                        str(len(val_loader)).zfill(3),
                        epochs_till_now,
                        final_epoch,
                        round(loss.item(), 5),
                        int(m),
                        round(s, 2))
                )

            batch_start_time = time.time()    

    GT_and_probs, roc_auc = get_roc_auc_score(gt, probs)

    with open(f'{model.__class__.__name__}.GT_and_probs', 'wb') as handle:
        pickle.dump(GT_and_probs, handle, protocol = pickle.HIGHEST_PROTOCOL)

    return val_loss_list, running_val_loss/float(len(val_loader.dataset)), roc_auc

def fit(device, train_loader, val_loader, model, model_name,
        loss_fn, optimizer, losses_dict, epochs_till_now, epochs,
        log_interval, save_interval):
    '''
    Trains or Tests the 'model' on the given 'train_loader', 'val_loader', 'test_loader' for 'epochs' number of epochs.
    If training ('test_only' = False), it saves the optimized 'model' and  the loss plots ,after every 'save_interval'th epoch.
    '''
    (epoch_train_loss,
     epoch_val_loss,
     total_train_loss_list,
     total_val_loss_list) = (losses_dict['epoch_train_loss'],
                             losses_dict['epoch_val_loss'],
                             losses_dict['total_train_loss_list'],
                             losses_dict['total_val_loss_list'])

    final_epoch = epochs_till_now + epochs

    print('\n======= Training after epoch #{}... =======\n'.format(epochs_till_now))

    best_roc_auc = 0.

    for epoch in range(epochs):
        epochs_till_now += 1
        print('============ EPOCH {}/{} ============'.format(epochs_till_now, final_epoch))
        epoch_start_time = time.time()

        print('TRAINING')
        train_loss, mean_running_train_loss = train_epoch(device,
                                                          train_loader,
                                                          model,
                                                          loss_fn,
                                                          optimizer,
                                                          epochs_till_now,
                                                          final_epoch,
                                                          log_interval)
        print('VALIDATION')
        val_loss, mean_running_val_loss, roc_auc = val_epoch(device,
                                                             val_loader,
                                                             model,
                                                             loss_fn,
                                                             epochs_till_now,
                                                             final_epoch,
                                                             log_interval)
        
        epoch_train_loss.append(mean_running_train_loss)
        epoch_val_loss.append(mean_running_val_loss)

        total_train_loss_list.extend(train_loss)
        total_val_loss_list.extend(val_loss)

        if (epoch +1) % save_interval == 0:
            save_path = os.path.join(config.models_dir, f'{model_name}.pth')
            ckpt = {
                'epochs': epochs_till_now,
                'model': model, # it saves the whole model
                'losses_dict': {'epoch_train_loss': epoch_train_loss,
                                'epoch_val_loss': epoch_val_loss,
                                'total_train_loss_list': total_train_loss_list,
                                'total_val_loss_list': total_val_loss_list}
            }
            torch.save(ckpt, save_path)

            print('\ncheckpoint {} saved'.format(save_path))

            if roc_auc > best_roc_auc:
                # save separate *best* checkpoint
                best_roc_auc = roc_auc
                save_path = os.path.join(config.models_dir, '{}.best.pth'.format(model_name))
                print(f"ROC AUC improved, saving 'best' checkpoint: {save_path}")
                torch.save(ckpt, save_path)

            fig = make_plot(epoch_train_loss,
                            epoch_val_loss,
                            total_train_loss_list,
                            total_val_loss_list)
            fig.savefig(os.path.join(config.models_dir,
                                     f'{model.__class__.__name__}.losses_{model_name}.png')
                        )
            print('loss plots saved !!!')

        print('\nTRAIN LOSS : {}'.format(mean_running_train_loss))
        print('VAL   LOSS : {}'.format(mean_running_val_loss))
        print('VAL ROC_AUC: {}'.format(roc_auc))

        total_epoch_time = time.time() - epoch_start_time
        m, s = divmod(total_epoch_time, 60)
        h, m = divmod(m, 60)
        print('\nEpoch {}/{} took {} h {} m'.format(epochs_till_now, final_epoch, int(h), int(m)))


def eval_(device, test_loader, model, loss_fn, log_interval):
    '''
    Tests the 'model' on the given 'test_loader'.
    '''
    print('\n======= Testing... =======\n')
    test_start_time = time.time()
    test_loss, mean_running_test_loss, test_roc_auc = val_epoch(device, test_loader, model, loss_fn, log_interval)
    total_test_time = time.time() - test_start_time
    m, s = divmod(total_test_time, 60)
    print('test_roc_auc: {} in {} mins {} secs'.format(test_roc_auc, int(m), int(s)))
    sys.exit()
