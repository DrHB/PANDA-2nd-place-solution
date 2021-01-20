import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pandas as pd
import time
from tqdm import tqdm
import apex.amp as amp
import albumentations as A
import os
import cv2

from dataset.dataset_patchlabel_smoothv2 import PANDA_Dataset_MultiTask_smooth_v2
from tools.utils import *
torch.backends.cudnn.benchmark = True
from sklearn.metrics import confusion_matrix,roc_auc_score
from tools.mixup import *
import matplotlib.pyplot as plt
import shutil
from radam import RAdam
def get_transforms_train():
    transforms=A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5,value=(255,255,255),mask_value=0),
        ]
    )
    return transforms

def get_transforms_train_patch():
    transforms=A.Compose(
        [
            A.OneOf([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],p=0.5
            ),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_REFLECT,p=0.2),
            A.RandomBrightnessContrast(p=0.25),
            #A.OneOf(
            #    [
            #
            #        A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.GaussianBlur(p=0.5),
            ],p=0.25
            ),
            A.CoarseDropout(max_holes=8,max_height=32,max_width=32,p=0.25,fill_value=(255,255,255))
        ]
    )
    return transforms

def log_cosh(pred,target,weight=None):
    if weight is not None:
        #print((torch.log(torch.cosh(pred - target))*weight))
        return 2 * (torch.log(torch.cosh(pred - target))*weight).mean()
    else:
        return 2*torch.log(torch.cosh(pred-target)).mean()

def mse(pred,target,weight=None):
    if weight is not None:
        #print((((target-pred)**2)*weight))
        return (((target-pred)**2)*weight).mean()
    else:
        return F.mse_loss(pred,target)

def huber(pred,target,weight=None):
    if weight is not None:
        return 2*(weight*F.smooth_l1_loss(pred,target,reduction='none')).mean()
    else:
        return 2*F.smooth_l1_loss(pred,target)
def smooth_cls(logits,target,pseudo_target,wt=0.7):
    b=logits.size(0)
    smooth_target=F.one_hot(target.long(),num_classes=6).float()
    #print(smooth_target)
    #print(pseudo_target)
    smooth_target=wt*smooth_target+pseudo_target*(1-wt)
    #print(smooth_target)
    #print("*"*50)
    log_prob=-F.log_softmax(logits,dim=1)
    return (log_prob*smooth_target).sum()/b

def sce_loss(logits,target,alpha=0.25,beta=0.25,eps=1e-6):
    b=logits.size(0)
    gt=F.one_hot(target.long(),num_classes=6).float()
    pred=F.softmax(logits,dim=1)

    gt=torch.clamp(gt,eps,1.0)
    pred=torch.clamp(pred,eps,1.0)

    log_gt=torch.log(gt)
    log_pred=torch.log(pred)


    ce=-(gt*log_pred).sum()/b
    rce=-(pred*log_gt).sum()/b

    return alpha*ce+beta*rce

def get_weight(module):
    for name,p in module.named_parameters():
        if "bias" not in name:
            yield p

def get_bias(module):
    for name,p in module.named_parameters():
        if "bias" in name:
            yield p


class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        shutil.copy(sys.argv[0],self.save_dir+sys.argv[0].split("/")[-1])
        df=pd.read_csv(options['train_csv_path'])


        suspicious=pd.read_csv(SETTINGS['SUSPICIOUS_PATH']).image_id
        df=df[~df.image_id.isin(suspicious)]

        if options['use_cleanlab']:
            df=df[df.noise_level!=1]

        train_df=df[df['fold']!=options['fold']]
        valid_df=df[df['fold']==options['fold']]

        train_dataset = PANDA_Dataset_MultiTask_smooth_v2(train_df,
                                         options['train_data_path'],
                                         options['train_mask_path'],
                                         num_patches=options['num_patches'],
                                         patch_size=options['patch_size'],
                                         augmentation=get_transforms_train(),
                                         augmentation_patch=get_transforms_train_patch(),
                                         deterministic=False,
                                         resample=False
                                         )
        valid_dataset = PANDA_Dataset_MultiTask_smooth_v2(valid_df,
                                         options['train_data_path'],
                                         options['train_mask_path'],
                                         num_patches=options['num_patches'],
                                         patch_size=options['patch_size'],
                                         deterministic=True,
                                         resample=False
                                         )
        self.valid_df=valid_df
        self.train_loader = DataLoader(train_dataset, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'],
                                       pin_memory=True, drop_last=True )
        self.valid_loader = DataLoader(valid_dataset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
                                       pin_memory=True, drop_last=False)

        self.batch_size = options['batch_size']
        self.num_train_img = len(train_dataset)
        self.num_valid_img = len(valid_dataset)
        print('Find %d train images, %d validation images' % (self.num_train_img, self.num_valid_img))
        print('Train batch size %d ,Validation batch size %d' % (options['batch_size'], options['batch_size']))

        self.device = options['device']
        self.model = model.to(self.device)

        param_groups=[
            {'params':get_weight(self.model),'weight_decay': 0.0002},
            {'params':get_bias(self.model),'weight_decay': 0},
        ]
        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(param_groups, lr=options['lr'], momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(param_groups, lr=options['lr'], betas=(0.9, 0.999))
        elif options['optimizer'] == 'radam':
            print("use RAdam")
            self.optimizer = RAdam(param_groups, lr=options['lr'], betas=(0.9, 0.999))
        else:
            raise NotImplementedError

        self.use_apex = options['use_apex']
        if options['use_apex']:
            opt_level = 'O1'
            self.model,self.optimizer=amp.initialize(self.model,self.optimizer,opt_level=opt_level)

        self.parallel = options['parallel']
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_QWK = -1
        self.max_epoch = options['max_epoch']
        save_parameters(options, self.save_dir + "parameters.txt")

        log_names=['Reg_Loss',"Cls_Loss",'Accuracy','QWK']
        self.log = {
            "train": {x: [] for x in log_names + ["patch_loss","patch_auc","epoch", "lr", "time"]},
            "valid": {x: [] for x in log_names + ["patch_loss","patch_auc","epoch", "time",'optimized_qwk',"kar","rad"]},
        }
        self.lam=options['lambda']

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        self.scheduler=ReduceLROnPlateau(self.optimizer,mode='max',factor=0.6,patience=2,verbose=True,threshold=0.0002,min_lr=1e-6)


        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")
    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_QWK = checkpoint['best_QWK']
        if self.parallel:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])


    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.module.state_dict() if self.parallel else self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_QWK": self.best_QWK
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv", index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: Train | Time: %s | Learning rate: %f" % (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()

        self.optimizer.zero_grad()
        total_loss_reg=0.
        total_loss_cls=0.
        total_loss_patch=0.
        all_prediction=[]
        all_groundtruth=[]
        patch_preds=[]
        patch_gt=[]

        lr_scheduler=None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.train_loader) - 1)
            lr_scheduler = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        for i, (image,label,patch_label,ignore_mask,pesudo_label,pesudo_prob,provider) in enumerate(tbar):
            #wt=-torch.clamp(torch.abs(label-pesudo_label),min=0.,max=3)/5.+1.
            #wt_kar=1
            #wt_rad=0.7
            #wt=wt_rad*provider+wt_kar*(1-provider)
            #wt=wt.cuda()
            wt=1.
            bs=image.size(0)
            image=image.to(self.device)
            label=label.to(self.device)

            reg_output,cls_output,patch_pred,A = self.model(image)
            reg_label=wt*label.float()+(1-wt)*pesudo_label.float().cuda()
            #loss_reg = F.mse_loss(reg_output,reg_label)
            loss_kar=mse(reg_output,reg_label,(1-provider).float().cuda())
            loss_rad=huber(reg_output,reg_label,provider.float().cuda())
            loss_reg=loss_kar+loss_rad
            #loss_cls = F.cross_entropy(cls_output,label.long())
            #loss_cls=sce_loss(cls_output,label.long())
            loss_cls=smooth_cls(cls_output,label.long(),pesudo_prob.float().cuda(),wt=0.7)
            loss_patch=F.binary_cross_entropy_with_logits(patch_pred.view(bs,-1),patch_label.cuda(),weight=ignore_mask.cuda())

            total_loss_reg+=loss_reg.item()
            total_loss_cls+=loss_cls.item()
            total_loss_patch+=loss_patch.item()

            loss=loss_reg+loss_patch+loss_cls
            loss /= self.accumulation_step

            valid_idx=np.where(ignore_mask.view(-1).detach().cpu().numpy()==1)
            patch_preds.append(torch.sigmoid(patch_pred.view(-1)).detach().cpu().numpy()[valid_idx])
            patch_gt.append(patch_label.view(-1).detach().cpu().numpy()[valid_idx])

            if self.use_apex:
                with amp.scale_loss(loss,self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()
                #print(self.optimizer.param_groups[0]['lr'])

            all_prediction.append(reg_output.detach().cpu().numpy())
            all_groundtruth.append(label.detach().cpu().numpy())
            tbar.set_description("Train loss: %.5f, cls loss: %.5f, Loss patch : %.5f" % (total_loss_reg/ (i + 1),
                                                                                          total_loss_cls/ (i + 1),
                                                                                          total_loss_patch/ (i + 1)))

        all_groundtruth=np.concatenate(all_groundtruth).reshape(-1)
        all_prediction=np.concatenate(all_prediction).reshape(-1)

        patch_preds=np.concatenate(patch_preds).reshape(-1)
        patch_gt=np.concatenate(patch_gt).reshape(-1)
        patch_auc=roc_auc_score(patch_gt,patch_preds)


        raw_prediction = get_prediction(all_prediction)

        accuracy=(raw_prediction==all_groundtruth).astype(np.float).sum()/len(all_groundtruth)

        qwk=qwk3(all_groundtruth,raw_prediction)
        self.log['train']['Reg_Loss'].append(total_loss_reg / num_batch)
        self.log['train']['Cls_Loss'].append(total_loss_cls / num_batch)
        self.log['train']['Accuracy'].append(accuracy)
        self.log['train']['QWK'].append(qwk)
        self.log['train']['patch_loss'].append(total_loss_patch / num_batch)
        self.log['train']['patch_auc'].append(patch_auc)


        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_train_img))
        print("Reg_Loss :{}".format(total_loss_reg / num_batch))
        print("Cls_Loss :{}".format(total_loss_cls / num_batch))

        print("Accuracy :{}".format(accuracy))
        print("QWK :{}".format(qwk))
        print("Patch BCE Loss: {}".format(total_loss_patch / num_batch))
        print("Patch AUC :{}".format(patch_auc))
        print()

    def valid(self, epoch):
        self.log['valid']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: Valid | Time: %s | Learning rate: %f" % (epoch, start, lr))
        self.log['valid']['time'].append(start)
        self.model.eval()

        total_loss_reg = 0.
        total_loss_cls=0.
        total_loss_patch =0.
        all_prediction = []
        all_groundtruth = []
        patch_preds=[]
        patch_gt=[]

        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (image, label,patch_label,ignore_mask,pesudo_label,pesudo_prob,provider) in enumerate(tbar):
            bs=image.size(0)
            image = image.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                reg_output, cls_output, patch_pred, A = self.model(image)
                loss_reg = F.mse_loss(reg_output, label.float())
                loss_cls = F.cross_entropy(cls_output, label.long())
                loss_patch = F.binary_cross_entropy_with_logits(patch_pred.view(bs, -1), patch_label.cuda(),
                                                                weight=ignore_mask.cuda())

                total_loss_reg += loss_reg.item()
                total_loss_cls += loss_cls.item()
                total_loss_patch += loss_patch.item()

            valid_idx=np.where(ignore_mask.view(-1).detach().cpu().numpy()==1)
            patch_preds.append(torch.sigmoid(patch_pred.view(-1)).detach().cpu().numpy()[valid_idx])
            patch_gt.append(patch_label.view(-1).detach().cpu().numpy()[valid_idx])

            all_prediction.append(reg_output.detach().cpu().numpy())
            all_groundtruth.append(label.detach().cpu().numpy())
            tbar.set_description("Valid loss: %.5f, cls loss: %.5f, Loss patch : %.5f" % (total_loss_reg/ (i + 1),
                                                                                          total_loss_cls/ (i + 1),
                                                                                          total_loss_patch/ (i + 1)))

        all_groundtruth = np.concatenate(all_groundtruth).reshape(-1)
        all_prediction = np.concatenate(all_prediction).reshape(-1)

        patch_preds=np.concatenate(patch_preds).reshape(-1)
        patch_gt=np.concatenate(patch_gt).reshape(-1)
        patch_auc=roc_auc_score(patch_gt,patch_preds)


        raw_prediction=get_prediction(all_prediction)
        round_optimizer=OptimizedRounder()
        round_optimizer.fit(all_prediction,all_groundtruth,initial_coef=[0.5,1.5,2.5,3.5,4.5])
        optimized_qwk=qwk3(round_optimizer.predict(all_prediction,round_optimizer.coefficients()),all_groundtruth)
        self.log['valid']['optimized_qwk'].append(optimized_qwk)
        optimized_prediction = round_optimizer.predict(all_prediction, round_optimizer.coefficients())

        accuracy = (raw_prediction == all_groundtruth).astype(np.float).sum() / len(all_groundtruth)
        qwk = qwk3(all_groundtruth, raw_prediction)

        self.log['valid']['Reg_Loss'].append(total_loss_reg / num_batch)
        self.log['valid']['Cls_Loss'].append(total_loss_cls / num_batch)
        self.log['valid']['Accuracy'].append(accuracy)
        self.log['valid']['QWK'].append(qwk)
        self.log['valid']['patch_loss'].append(total_loss_patch / num_batch)
        self.log['valid']['patch_auc'].append(patch_auc)

        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_valid_img))
        print("Reg_Loss :{}".format(total_loss_reg / num_batch))
        print("Cls_Loss :{}".format(total_loss_cls / num_batch))
        print("Accuracy :{}".format(accuracy))
        print("QWK :{}".format(qwk))
        #self.scheduler.step(qwk)
        if True:
            provider = [x == "karolinska" for x in self.valid_df.data_provider]
            provider = np.array(provider).astype(np.int)
            print("karolinska")
            idx = np.where(provider == 1)
            self.log['valid']['kar'].append(qwk3(optimized_prediction[idx], all_groundtruth[idx]))
            print("QWK:", qwk3(optimized_prediction[idx], all_groundtruth[idx]))
            print(confusion_matrix(all_groundtruth[idx], optimized_prediction[idx]))

            print("radboud")
            idx = np.where(provider == 0)
            self.log['valid']['rad'].append(qwk3(optimized_prediction[idx], all_groundtruth[idx]))
            print("QWK:", qwk3(optimized_prediction[idx], all_groundtruth[idx]))

            print(confusion_matrix(all_groundtruth[idx], optimized_prediction[idx]))
            print("Optimized QWK :{}".format(optimized_qwk))
            print("Overall confusion matrix")
            print(confusion_matrix(all_groundtruth,optimized_prediction))
        print()

        print("Patch BCE Loss: {}".format(total_loss_patch / num_batch))
        print("Patch AUC :{}".format(patch_auc))

        if optimized_qwk > self.best_QWK:
            print("Find a better model with QWK, {} -> {}".format(self.best_QWK,  optimized_qwk ))
            self.best_QWK = optimized_qwk
            self.save_checkpoint(epoch, save_optimizer=False, suffix="bestOptimQWK")
        else:
            print("This model {}, best model {}".format( optimized_qwk , self.best_QWK))





    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch+5):
            if epoch in self.lr_step:
                for i in range(len(self.optimizer.param_groups)):
                    self.optimizer.param_groups[i]['lr']*=self.lr_decay_ratio
            self.train(epoch)
            self.valid(epoch)
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            if epoch % 5 == 4:
                self.save_checkpoint(epoch, save_optimizer=False, suffix="epoch" + str(epoch))
            self.save_log()

import json
with open("./SETTINGS.json") as f:
    SETTINGS=json.load(f)

def main(fold):
    options=dict()
    options['train_data_path'] = SETTINGS['LEVEL1_IMAGE_DIR']
    options['train_mask_path'] = SETTINGS['LEVEL1_MASK_DIR']
    options['train_csv_path'] = SETTINGS['CSV_PATH']
    options['fold']=int(fold)

    options['device'] = "cuda"
    options['use_apex'] =True
    options['parallel']= False

    options['use_cleanlab']=False

    options['batch_size'] =10
    options['accumulation_step']=3
    options['num_workers']=10

    options['lambda']=1
    options['optimizer'] = 'adam'
    options['lr'] = 0.0005

    options['max_epoch'] = 50

    options['lr_step'] = (20,35,45)
    options['lr_decay_ratio']=0.2

    options['resume_path']=None

    options['arch']="se_resnext50_32x4d"
    options['model_name']="{}_fold{}".format(options['arch'],fold)
    options['save_dir'] = "./result_efb0/{}_regression_level1_msehuber_fold{}/".format(options['arch'],fold)
    options['dropout']=0.25
    options['regression']=True

    options['patch_size']=192
    options['num_patches']=48

    from models import PANDA_Model_Attention_Concat_MultiTask_Headv2
    model=PANDA_Model_Attention_Concat_MultiTask_Headv2(arch=options['arch'],
                                                 dropout=options['dropout'],
                                                 num_classes=6,
                                                 checkpoint=True,
                                                scale_op=True,
                                                 )
    print(model)

    trainer = Trainer(model, options)
    print("Start Trainning fold",fold)
    print("Save Result to",options['save_dir'])
    trainer.start_train()


if __name__ == "__main__":
    import sys
    fold=int(sys.argv[1])
    main(fold)
