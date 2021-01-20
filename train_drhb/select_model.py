import os 
import pretrainedmodels
from pdb import set_trace
import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback, CSVLogger
import os
from sklearn.model_selection import StratifiedKFold
import timm
from sklearn.metrics import cohen_kappa_score,confusion_matrix
import albumentations as A
import json
import argparse
import cv2
from tqdm import tqdm
import cv2
PIL.Image.MAX_IMAGE_PIXELS = None
import scipy as sp
import argparse

with open('SETTINGS.json', 'r') as fp: settings = json.load(fp)
FOLD = settings['FOLD']
TRAIN_PREPROC = Path(settings['CLEAN_DATA_DIR'])
CUSTOM_DATA = Path(settings['CUSTOM_DATA'])


#reading dataframe and removing bad images
if str(CUSTOM_DATA) == '.':
    df = pd.read_csv(settings['TRAIN_DATA_CLEAN_PATH'])
    #insuring that all the images present for training that are preproccess
    df_process = pd.DataFrame({'image_id' :list(set([i.split('_')[0] for i in os.listdir(TRAIN_PREPROC)]))}) 
    df = pd.merge(df, df_process, on='image_id')

    
else:
    df = pd.read_csv(CUSTOM_DATA/'MASTER_PANDAS.csv')
    df_corupt = pd.read_csv(CUSTOM_DATA/'PANDA_Suspicious_Slides.csv')
    df_corupt.set_index('image_id', inplace = True)
    df.set_index('image_id', inplace=True)
    for i in df_corupt.index.values:
        try:
            df.drop(i, inplace=True)
        except: 
            pass
    
    df.reset_index(inplace=True)
    df_process = pd.DataFrame({'image_id' : list(set([i.split('_')[0] for i in os.listdir(TRAIN_PREPROC)]))}) 
    df = pd.merge(df, df_process, on='image_id')
    
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




class Model(nn.Module):
    def __init__(self, N ):
        super().__init__()
        m = models.resnet34(pretrained=True)
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
        
        self.N=N
        
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
    
    
class PandasDSST2(Dataset):
    def __init__(self, fnames, stats, N, sz):
      self.items = fnames 
      self.stats = list(map(tensor, stats))
      self.sz = sz
      self.N =N
      self.fn_dict = {i: [f'{str(TRAIN_PREPROC)}/{i}_{k}.png' for k in range(N)] for i in fnames}

    def __len__(self): 
      return len(self.items)

    def __getitem__(self, idx):
        imgs = []
        fns = self.fn_dict[self.items[idx]]
        for fn in fns:
            img = PIL.Image.open(fn).convert('RGB')
            img = tensor(np.array(img)).float()/255
            img = self.normalize(img)
            imgs.append(img)
        return torch.cat(imgs).reshape(self.N, self.sz, self.sz, 3).permute(0, 3, 1, 2)

    def normalize(self, x):
      return (x-self.stats[0])/self.stats[1]


def get_scores(t, p):
    scr = cohen_kappa_score(t, p,weights='quadratic')
    print(scr)
    return scr
    
    
def get_preds(tta, md_path, N, sz, NFOLDS=1):
    score_dict = dict()
    score_dict['model_path'] = md_path
    pred = []
    bs = 2
    sz = 224
    for FOLD in range(NFOLDS):
        print (FOLD)
        fns = df[df['split']==FOLD]['image_id'].to_list()
        pandas_dl = DataLoader(PandasDSST2(fns, imagenet_stats, N, sz), batch_size=bs, num_workers=10)
        model_path =md_path
        print (f'loading: {model_path}')
        state_dict = torch.load(model_path,map_location=torch.device('cpu'))['model']
        model = Model(N)
        model.load_state_dict(state_dict)
        model.float()
        model.cuda()
        model.eval()
        with torch.no_grad():
            if tta:
                for x in tqdm(pandas_dl):
                    x = x.cuda()
                    x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
                      x.transpose(-1,-2),x.transpose(-1,-2).flip(-1),
                      x.transpose(-1,-2).flip(-2),x.transpose(-1,-2).flip(-1,-2)],1)
                    x = x.view(-1,N,3,sz,sz)   
                    p = model(x)
                    try:
                        p = p.view(bs, 8 ,-1).mean(1).cpu()
                    except:
                        p = p.view(1, 8 ,-1).mean(1).cpu()
                    pred.append(p.float().cpu())
            else:
                for x in tqdm(pandas_dl):
                    x = x.cuda()
                    p = model(x)
                    pred.append(p.float().cpu())
                    
    optR = OptimizedRounder()
    coefficients = [0.5, 1.5, 2.5, 3.5, 4.5]
    
    t = [df[df['split']==i]['isup_grade'].values for i in range(NFOLDS)]
    t = np.concatenate(t, axis=0)

    data_provider = [df[df['split']==i]['data_provider'].values for i in range(NFOLDS)]
    data_provider = np.concatenate(data_provider, axis=0)

    image_id = [df[df['split']==i]['image_id'].values for i in range(NFOLDS)]
    image_id = np.concatenate(image_id, axis=0)

    p = np.concatenate(pred, axis=0).reshape(-1)
    df_res = pd.DataFrame({'t': t, 'p_raw': p, 'data_provider': data_provider, 'image_id': image_id})
    df_res['p_standart'] = optR.predict(p, coefficients)
    print ('_______________')
    print ('SCORE OVERALL:')
    score_dict['overall'] = get_scores(df_res['t'], df_res['p_standart'])
    print ('SCORE KAROLINSKA:')
    score_dict['karolinska'] = get_scores(df_res[df_res['data_provider'] == 'karolinska']['t'], df_res[df_res['data_provider'] == 'karolinska']['p_standart'],)
    print ('SCORE RADBOUND:')
    score_dict['radbound'] = get_scores(df_res[df_res['data_provider'] == 'radboud']['t'], df_res[df_res['data_provider'] == 'radboud']['p_standart'],)
    return score_dict



def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--phase", 
        default=1, 
        type=int, 
        help="Training Phase 1 or 2"
    )
    
    
    parser.add_argument(
        "--epoch", 
        default=0, 
        type=int, 
        help="epoch of training between 0 to 59"
    )

    args = parser.parse_args()
    SZ =224
    
    if args.phase == 1:
        NAME = 'EXP_80'
        SUFFIX =f'EXP_80_RESNET_34_TILES_49_SQ_FT_NBN_SE_DUAL_0_sq_features'
        N = 49
        
        
    if args.phase == 2:
        NAME = 'EXP_80'
        SUFFIX =f'EXP_80_RESNET_34_TILES_81_SQ_FT_NBN_SE_DUAL_0_sq_features'
        N=81
        
    chkp = f"{settings['MODEL_CHECKPOINT_DIR']}/{SUFFIX}_{args.epoch}.pth"
    print (f"TILES: {N}")
    print (f"EPOCH: {args.epoch}")
    print (f"CHKP : {chkp}") 
    res = get_preds(True, chkp, N, SZ)
    pd.DataFrame(res, index=[0]).to_csv(f"{settings['LOGS_DIR']}/TTA_{SUFFIX}_{args.epoch}.csv", index=False)
    

    
if __name__=='__main__':
    main()
 

    

    