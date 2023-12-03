import os
import random
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 28#root_config['random_seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import csv
import yaml
import shutil
import sys
import dill
import time
import warnings
import numpy as np
from random import sample
from sklearn import metrics
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset.data_finetune import CIFData
from dataset.data_finetune import collate_pool, get_train_val_test_loader
from model.en_de_extend import finetune_ENDE
from dataset.graph import *
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./finetune_config/config_ft.yaml', os.path.join(model_checkpoints_folder, 'config_ft.yaml'))


def _check_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)


class FineTune(object):
    def __init__(self, config, root_config, target_dataset, i_num, current_time):
        self.config = config
        self.root_config = root_config
        self.i_num = i_num
        self.device = self._get_device()
        dir_name = current_time
        log_dir = os.path.join('runs_aug', dir_name)
        _check_file(log_dir)
        log_dir_num =  os.path.join(log_dir, target_dataset)
        _check_file(log_dir_num)

        self.writer = SummaryWriter(log_dir=log_dir_num)

        if self.config['task'] == 'classification':
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.L1Loss()

        #self.dataset_train = CIFData('train', i_num,  self.config['task'], **self.config['dataset'])
        #self.dataset_val   =  CIFData('val', i_num, self.config['task'], **self.config['dataset'])
        #self.dataset_test  =  CIFData('test', i_num, self.config['task'], **self.config['dataset'])
        #self.target_dataset = target_dataset
        #self.random_seed = self.config['random_seed']
        #collate_fn = collate_pool
        #self.train_loader, self.valid_loader, self.test_loader = get_train_val_test_loader(
        #    dataset_train = self.dataset_train,
        #    dataset_val = self.dataset_val,
        #    dataset_test = self.dataset_test,
        #    collate_fn = collate_fn,
        #    pin_memory = self.config['cuda'],
        #    batch_size = self.root_config['batch_size'], 
        #    return_test = True,
        #    **self.root_config['dataloader']
        #)

        data_loader_root = 'Dataset/'
        _check_file(data_loader_root)
        data_loader_path = data_loader_root+target_dataset
        if  os.path.exists(data_loader_path):
             with open(data_loader_path+'/train.pkl','rb') as f:
                   self.train_loader = dill.load(f)
             with open(data_loader_path+'/val.pkl','rb') as f:
                   self.valid_loader = dill.load(f)
             with open(data_loader_path+'/test.pkl','rb') as f:
                   self.test_loader = dill.load(f)
             with open(data_loader_path+'/sample.pkl', 'rb') as f:
                   sample_target = dill.load(f)

        else:
            self.dataset_train = CIFData('train', i_num,  self.config['task'], **self.config['dataset'])
            self.dataset_val   =  CIFData('val', i_num, self.config['task'], **self.config['dataset'])
            self.dataset_test  =  CIFData('test', i_num, self.config['task'], **self.config['dataset'])
            self.target_dataset = target_dataset
            sample_target = torch.tensor(self.dataset_train.id_goal+self.dataset_val.id_goal_val)
            self.random_seed = self.config['random_seed']
            collate_fn = collate_pool
            self.train_loader, self.valid_loader, self.test_loader = get_train_val_test_loader(dataset_train = self.dataset_train, dataset_val = self.dataset_val,dataset_test = self.dataset_test,collate_fn = collate_fn,pin_memory = self.config['cuda'],batch_size = self.root_config['batch_size'],return_test = True,**self.root_config['dataloader'])
            _check_file(data_loader_path)
            with open(data_loader_path+'/train.pkl','wb') as f:
                  dill.dump(self.train_loader, f)
            with open(data_loader_path+'/val.pkl','wb') as f:
                  dill.dump(self.valid_loader, f)
            with open(data_loader_path+'/test.pkl','wb') as f:
                  dill.dump(self.test_loader, f)
            with open(data_loader_path+'/sample.pkl','wb') as f:
                  dill.dump(sample_target, f)







        # obtain target value normalizer
        #if self.config['task'] == 'classification':
        #    self.normalizer = Normalizer(torch.zeros(2))
        #    self.normalizer.load_state_dict({'mean': 0., 'std': 1.})
        #else:
        #    sample_target = torch.tensor(self.dataset_train.id_goal+self.dataset_val.id_goal_val)
        self.normalizer = Normalizer(sample_target) 


    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device

    def group_decay(self, model):
        """Omit weight decay from bias and batchnorm params."""
        decay, no_decay = [], []

        for name, p in model.named_parameters():
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        return [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0},
            ]


    def train(self, cutoff, neibor, loss_weight, dynamic_wieght):
        
        model = finetune_ENDE(cutoff, neibor)
        print("model_loading........")  
        if self.config['cuda']:
            model = model.to(self.device)
        
        layer_list = []
        for name, param in model.named_parameters():
            if 'extend' in name:
                print(name, 'new layer')
                layer_list.append(name)
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))


        #params = self.group_decay(model)
        
        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.config['optim']['lr']*0.2}, {'params': params}],
                 self.config['optim']['lr'], momentum=self.config['optim']['momentum'], 
                weight_decay=self.config['optim']['weight_decay'])
            
        elif self.config['optim']['optimizer'] == 'Adam':
            optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=self.root_config['epochs'],
            steps_per_epoch=len(self.train_loader),)#20

        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0
        weights = torch.tensor([1,0])
        decay_rate = 0.5
        decay_iter = 0.5
        loss_history = []

        for epoch_counter in range(self.root_config['epochs']):
            model.train()
            for bn, (input_1, input_2) in enumerate(self.train_loader):
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(input_1.target)
                else:
                    target_normed = target.view(-1).long()
                
                target_var = target_normed.cuda()

                # compute output
                output,target_var,loss, test_out, loss_force = model(input_1, input_2, target_var, 'train', dynamic_wieght)

                #output=output[:,0]
                # print(output.shape, target_var.shape)
                optimizer.zero_grad()
                loss1 = self.criterion(output, target_var)
                loss2 = weights[0] *loss1 + weights[1]*loss#+#loss_weight*0.1*loss_force
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss1.item(), global_step=n_iter)
                    print(epoch_counter, bn, loss1.item())
                
                loss2.backward()
                optimizer.step()
                print('TRAIN INFO: epoch:{} ({}/{}) iter:{} con_loss:{:.5f} mae_loss:{:.5f} force_loss:{:.5f} loss:{:.5f}'.format(epoch_counter, bn + 1, len(self.train_loader), n_iter, loss.item(), loss1.item(), loss_force.item(),loss2.item()))
                n_iter += 1

            if epoch_counter  == 50:
                #decay_iter *= decay_rate
                #if  decay_iter < 0.01:
                #    weights[1] = 0
                #weights[0]
                weights = torch.tensor([1, 0])
                for name, param in model.named_parameters():
                    if 'extend' in name:
                        param.requires_grad = False
            print(weights[0],weights[1])

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['task'] == 'classification': 
                    valid_loss, valid_roc_auc, loss_con = self._validate(model, self.criterion, self.valid_loader, dynamic_wieght)
                    if valid_roc_auc > best_valid_roc_auc:
                        # save the model weights
                        best_valid_roc_auc = valid_roc_auc
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['task'] == 'regression': 
                    valid_loss ,valid_mae, loss_con = self._validate(model, self.criterion, self.valid_loader,dynamic_wieght )
                    if valid_mae  < best_valid_mae:
                    #if loss_con  < best_valid_mae:
                        # save the model weights
                        best_valid_mae = valid_mae
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                    self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                    valid_n_iter += 1
            scheduler.step()
        self.model = model
        

    def _test_load(self, epoch_num, cutoff, neigbor):
        loss, metric, metric_rmse, pred, target = self.test(cutoff, neigbor)
        import pandas as pd
        ftf = root_config['fine_tune_from'].split('/')[-1]
        seed = root_config['random_seed']
        fn = '{}_{}_nofine_{}.csv'.format(ftf, task_name,target_dataset)
        print(fn)
        titles= ['num'+str(epoch_num), 'loss', 'mae','rmse']
        df = pd.DataFrame([[i_num, loss, metric.item(), metric_rmse.item()]], columns=titles)
        df.to_csv(os.path.join('experiments', fn), mode='a', index=False)
        df_result = pd.DataFrame(data = {'target': target, 'pred': pred})
        fn_result = os.path.join('experiments', target_dataset)
        _check_file(fn_result)
        df_result.to_csv(os.path.join(fn_result, str(i_num)+fn),  index=False)

    def _load_pre_trained_weights(self, model):
        try:
            target_task = root_config["target_dataset"]
            checkpoints_folder = os.path.join(self.root_config['fine_tune_from']+'/'+target_task, 'checkpoints')
            load_state = torch.load(os.path.join(checkpoints_folder, 'model.pth'),  map_location=self.config['gpu']) 
 
            model_state = model.state_dict()
            for name, param in load_state.items():
                if name not in model_state:
                    print('NOT loaded:', name)
                    continue
                else:
                    print('loaded:', name)
                if isinstance(param, nn.parameter.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                model_state[name].copy_(param)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _validate(self, model, criterion, valid_loader, dynamic_coefficient):
        losses = AverageMeter()
        if self.config['task'] == 'regression':
            mae_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()
        model.eval()
        total_loss = []
        val_targets = []
        val_preds = []
        with torch.no_grad():
            #model.eval()

            for bn, (input1, input2) in enumerate(valid_loader):
                target = input1.target 
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()
                
                target_var = target_normed.cuda()

                # compute output
                output, target_var, loss, test_out,loss_force = model(input1, input2, target_var, 'val', dynamic_coefficient)
                #output=output[:,0]
                #import ipdb
                #ipdb.set_trace()/
                output = output[:self.root_config['batch_size']]
                target_var = target_var[: self.root_config['batch_size']]
                loss1 = criterion(output, target_var)

                val_pred = self.normalizer.denorm(output.data.cpu())
                val_target = self.normalizer.denorm(target_var.data.cpu())

                val_preds += val_pred.view(-1).tolist()
                val_targets += val_target.view(-1).tolist()
                
                mae_error = mae(self.normalizer.denorm(output.data.cpu()),  self.normalizer.denorm(target_var.data.cpu()))
                mae_errors.update(mae_error, target_var.size(0))
                total_loss.append(loss1.item())           
            
            total_loss = sum(total_loss)/len(total_loss)
            mae_result = np.sum(np.abs(np.array(val_preds)-np.array(val_targets)))/len(val_preds)
            #if self.config['task'] == 'regression':
            print('MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(mae_errors=mae_errors))
            print(mae_result)
            print(total_loss)
        
        model.train()

        if self.config['task'] == 'regression':
            #print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
            return mae_result, mae_result, loss1#, loss, losstotal_loss, total_loss#losses.avg, mae_errors.avg
        else:
            print('AUC {auc.avg:.3f}'.format(auc=auc_scores))
            return losses.avg, auc_scores.avg, loss

    
    def test(self, cutoff, neibor):
        test_model = finetune_ENDE(cutoff, neibor).cuda()
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        print(model_path)
        
        state_dict = torch.load(model_path, map_location=self.device)
        test_model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        losses = AverageMeter()
        if self.config['task'] == 'regression':
            mae_errors = AverageMeter()
            rmse_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()
        
        test_targets = []
        test_preds = []
        test_cif_ids = []

        with torch.no_grad():
            test_model.eval()
            for bn, (input1, input2) in enumerate(self.test_loader):
                target = input1.target
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()
                
                target_var = target_normed.cuda()

                # compute output
                output, target_, loss, test_out,loss_force = test_model(input1, input2, target_var, 'test', 0)
                #output=output[:,0]
                loss1 = self.criterion(test_out, target_var)

                if self.config['task'] == 'regression':
                    mae_error = mae(self.normalizer.denorm(test_out.data.cpu()), target)
                    rmse_error = rmse(self.normalizer.denorm(test_out.data.cpu()), target)
                    #losses.update(loss.data.cpu().item(), target.size(0))
                    mae_errors.update(mae_error, target.size(0))
                    rmse_errors.update(rmse_error, target.size(0))
                    test_pred = self.normalizer.denorm(test_out.data.cpu())
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += input1.cif_id
                else:
                    accuracy, precision, recall, fscore, auc_score = \
                        class_eval(output.data.cpu(), target)
                    losses.update(loss.data.cpu().item(), target.size(0))
                    accuracies.update(accuracy, target.size(0))
                    precisions.update(precision, target.size(0))
                    recalls.update(recall, target.size(0))
                    fscores.update(fscore, target.size(0))
                    auc_scores.update(auc_score, target.size(0))
                   
                    test_pred = torch.exp(output.data.cpu())
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += input1.cif_id

            mae_result = np.sum(np.abs(np.array(test_preds)-np.array(test_targets)))/len(test_preds)
            rmse_result = np.sqrt( np.sum(np.abs(np.array(test_preds)-np.array(test_targets))**2)/len(test_preds))
            if self.config['task'] == 'regression':
                print('Test: [{0}/{1}], ''Loss {loss.val:.4f} ({loss.avg:.4f}), ''MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    bn, len(self.valid_loader), loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}], '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Accu {accu.val:.3f} ({accu.avg:.3f}), '
                      'Precision {prec.val:.3f} ({prec.avg:.3f}), '
                      'Recall {recall.val:.3f} ({recall.avg:.3f}), '
                      'F1 {f1.val:.3f} ({f1.avg:.3f}), '
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    bn, len(self.valid_loader), loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

        with open(os.path.join(self.writer.log_dir, 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
        
        #self.model.train()

        if self.config['task'] == 'regression':
            print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
            return loss, mae_result, rmse_result, np.array(test_preds), np.array(test_targets)
        else:
            print('AUC {auc.avg:.3f}'.format(auc=auc_scores))
            return losses.avg, auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class Normalizer_maxmin(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.max = torch.max(tensor)
        self.min = torch.min(tensor)

    def norm(self, tensor):
        return (tensor - self.min) /(self.max-self.min)

    def denorm(self, normed_tensor):
        return normed_tensor * (self.max-self.min) + self.min

    def state_dict(self):
        return {'max': self.max,
                'min': self.min}

    def load_state_dict(self, state_dict):
        self.max = state_dict['max']
        self.min = state_dict['min']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))

def rmse(prediction, target):
    return torch.mean(torch.abs(target - prediction)**2)

def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

if __name__ == "__main__": 
  root_config = yaml.load(open("finetune_config/config_ft.yaml", "r"), Loader=yaml.FullLoader)
    
  target_dataset = root_config["target_dataset"]
  iter_num = root_config['iter_num']
  if target_dataset == 'matbench_dielectric': 
        config = yaml.load(open("finetune_config/config_mb_dielect.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'matbench_jdft2d':
        config = yaml.load(open("finetune_config/config_mb_jdft2d.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset =='matbench_log_kvrh':
        config = yaml.load(open("finetune_config/config_mb_logkv.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset =='matbench_mp_gap':
        config = yaml.load(open("finetune_config/config_mp_gap.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'mp_gap':
        config = yaml.load(open("finetune_config/config_mp_gap.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'mp_bulk':
        config = yaml.load(open("finetune_config/config_mp_bulk.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'mp_shear':
        config = yaml.load(open("finetune_config/config_mp_shear.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'jarvis_gap_mbj':
        config = yaml.load(open("finetune_config/config_jarvis_gap_mbj.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'jarvis_ehull':
        config = yaml.load(open("finetune_config/config_jarvis_ehull.yaml", "r"), Loader=yaml.FullLoader)
  else:
        config = root_config

  print(config)

  config['task'] =  'regression'
  task_name = target_dataset
  for loss_weight in [0.01]:
   for dynamic_wieght in [0.7]:

    metric_list = []
    rmse_list = []
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    titles= ['num', 'loss', 'mae','rmse']

    i_num = 0
    cutoff = 8.0
    neibor  = 12
    fine_tune = FineTune(config, root_config, target_dataset, i_num, current_time)
    fine_tune.train(cutoff, neibor, loss_weight, dynamic_wieght)
    loss, metric, metric_rmse, pred, target = fine_tune.test(cutoff, neibor)
    import pandas as pd
    ftf = root_config['fine_tune_from'].split('/')[-1]
        #seed = root_config['random_seed']
    fn = '{}_{}_nofine_{}.csv'.format(ftf, task_name,target_dataset)
    print(fn)
    metric_list.append(metric.item())
    rmse_list.append(metric_rmse.item())
    df = pd.DataFrame([[i_num, loss, metric.item(), metric_rmse.item()]], columns=titles)
    df.to_csv(
            os.path.join('experiments', fn),
            mode='a', index=False
        )
    df_result = pd.DataFrame(data = {'target': target, 'pred': pred})
    fn_result = os.path.join('experiments', target_dataset)
    _check_file(fn_result)
    df_result.to_csv(os.path.join(fn_result, str(i_num)+fn),  index=False)

    import numpy as np
    df = pd.DataFrame([[neibor, str(current_time)+'|_|'+str(loss_weight)+'|_|'+str(dynamic_wieght), np.mean(metric_list),np.mean(rmse_list)]], columns=titles)
    df.to_csv(
            os.path.join('experiments', fn),
            mode='a', index=False)
