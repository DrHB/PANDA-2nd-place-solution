import os
import cv2
import skimage.io
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import PIL
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch import nn
import argparse
import fastai
from fastai.vision import *
from joblib import Parallel, delayed

class CustomEnd(nn.Module):
    def __init__(self, scaler = SigmoidRange(-1, 6.0)):
        super().__init__()
        self.scaler_ = scaler

    def forward(self, x):
        classif = x[:, :-1]
        regress = self.scaler_ (x[:, -1])
        return classif, regress




def make_divisible(v, divisor=8, min_value=None):
   min_value = min_value or divisor
   new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
   # Make sure that round down does not go down by more than 10%.
   if new_v < 0.9 * v:
      new_v += divisor
   return new_v
def sigmoid(x, inplace: bool = False):
   return x.sigmoid_() if inplace else x.sigmoid()
class SqueezeExcite(nn.Module):
   def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
             act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
      super(SqueezeExcite, self).__init__()
      self.gate_fn = gate_fn
      reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
      self.act1 = act_layer(inplace=True)
      self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
   def forward(self, x):
      x_se = self.avg_pool(x)
      x_se = self.conv_reduce(x_se)
      x_se = self.act1(x_se)
      x_se = self.conv_expand(x_se)
      x = x * self.gate_fn(x_se)
      return x

class CustomEnd(nn.Module):
    def __init__(self, scaler = SigmoidRange(-1, 6.0)):
        super().__init__()
        self.scaler_ = scaler

    def forward(self, x):
        classif = x[:, :-1]
        regress = self.scaler_ (x[:, -1])
        return classif, regress


class ModelSeDul(nn.Module):
    def __init__(self, N):
        super().__init__()
        m = fastai.vision.models.resnet34()
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.cb = SqueezeExcite(nc)
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(2*nc,512),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(0.4),
                                  nn.Linear(512,7), 
                                  CustomEnd())
        
        self.N = N

    def forward(self, x):
        shape = x.shape
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        x = self.enc(x)
        shape = x.shape
        x = x.view(-1, n, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4).contiguous().\
        view(-1, x.shape[1], x.shape[2] * n, x.shape[3])
        x = x.view(x.shape[0], x.shape[1], x.shape[2]//int(np.sqrt(self.N)), -1)
        x = self.cb(x)
        x = self.head(x)
        return x[1]


class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']

    
        

def main(debug=False):
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        default='EXP_80_RESNET_34_TILES_81_SQ_FT_NBN_SE_DUAL_0_sq_features_41_best.pth', 
        type=str, 
        help="Number of cpu to use to process data"
    )
    args = parser.parse_args()
    with open('SETTINGS.json', 'r') as fp: settings = json.load(fp)
    
    N_FOLDS = 1
    sz = 224
    bs = 2
    N = 81
    nworkers = 2
    imagenet_stats  = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    print (f'getting predictions using: {args.model_name}')
    DATA = settings["RAW_DATA_TEST"]
    TEST = settings['TEST_DATA_CLEAN_PATH']
    IMG_OUT = 'imgs'
    MODELS = [f"{settings['MODEL_CHECKPOINT_DIR']}/{args.model_name}"]


    def prepare_df(df):
        uniq_id = list(df['image_id'].unique())
        return df, uniq_id


    def tile(img_name):
        img = skimage.io.MultiImage(img_name)[-2]
        result = []
        shape = img.shape
        pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                    constant_values=255)

        img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)


        img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
        idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))
        img = img[idxs][:N]

        if len(img)<N:
            n = N-len(img)
            img = np.concatenate([img, np.full((n,sz,sz,3),255,dtype=np.uint8)],0)

        for i in range(len(img)):
            result.append({'img':img[i],  'idx':i})
        return result



    def save_imgs(image_name, folder_path=DATA, image_folder_out=IMG_OUT):
        image_nm = folder_path + '/' + image_name + '.tiff' 
        try:
            tiles = tile(image_nm)
            for t in tiles:
                img,idx = t['img'], t['idx']
                cv2.imwrite(f'{image_folder_out}/{image_name}_{idx}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        except:
            pass






    class PandasDSST2(Dataset):
        def __init__(self, fnames,  stats=imagenet_stats, N=N, sz=sz, path = IMG_OUT):
          self.items = fnames 
          self.stats = list(map(tensor, stats))
          self.sz = sz
          self.N =N
          self.path = path

        def __len__(self): 
          return len(self.items)

        def __getitem__(self, idx):
            imgs = []
            fns = [f'{self.path}/{self.items[idx]}_{i}.png'   for i in range(self.N)]
            for fn in fns:
                img = PIL.Image.open(fn).convert('RGB')
                img = tensor(np.array(img)).float()/255
                img = self.normalize(img)
                imgs.append(img)
            return torch.cat(imgs).reshape(self.N, self.sz, self.sz, 3).permute(0, 3, 1, 2), self.items[idx]

        def normalize(self, x):
          return (x-self.stats[0])/self.stats[1]



    models = []
    for path in MODELS:
        state_dict = torch.load(path,map_location=torch.device('cpu'))['model']
        model = ModelSeDul(N)
        model.load_state_dict(state_dict)
        model.float()
        model.eval()
        model.cuda()
        models.append(model)


    optR = OptimizedRounder()
    coefficients = [0.5, 1.5, 2.5,  3.5, 4.5]


    def get_preds(N, nm, md_, TTA=False):
        print('getting predictions:')
        print (f"{N}")
        ds = PandasDSST2(nm, N=N)
        dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)
        names,preds = [],[]
        if TTA:
            with torch.no_grad():
                for x,y in tqdm(dl):
                    x = x.cuda()
                    x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
                      x.transpose(-1,-2),x.transpose(-1,-2).flip(-1),
                      x.transpose(-1,-2).flip(-2),x.transpose(-1,-2).flip(-1,-2)],1)
                    x = x.view(-1,N,3,sz,sz)            
                    p = [model(x) for model in md_]
                    p = torch.stack(p,1)
                    try:
                        p = p.view(bs, 8*len(md_),-1).mean(1).cpu()
                    except: 
                        bs_= p.shape[0]
                        p = p.view(bs_, 8*len(md_),-1).mean(1).cpu()
                    names.append(y)
                    preds.append(p) 

        else:
            with torch.no_grad():
                for x,y in tqdm(dl):
                    x = x.cuda()
                    x = x.view(-1,N,3,sz,sz)
                    p = [model(x) for model in md_]
                    p = torch.stack(p,1)
                    try:
                        p = p.view(bs, len(md_),-1).mean(1).cpu()
                    except:
                        bs_= p.shape[0]
                        p = p.view(bs_, len(md_),-1).mean(1).cpu()
                    names.append(y)
                    preds.append(p)

        names = np.concatenate(names)
        preds = torch.cat(preds).numpy().reshape(-1)
        pred_ = optR.predict(preds, coefficients).astype('int32')
        sub_df = pd.DataFrame({'image_id': names, 'isup_grade': preds})
        return sub_df


       
    nm = pd.read_csv(TEST).image_id.to_list()
    os.mkdir('imgs')
    Parallel(n_jobs=nworkers)(delayed(save_imgs)(i) for i in tqdm(nm));
    sub_df = get_preds(81, nm, models, TTA=True)
    sub_df.to_csv(f"{settings['SUBMISSION_DIR']}/submission.csv", index=False)
    os.system("rm -rf imgs" )

    
if __name__=='__main__':
    main()
 