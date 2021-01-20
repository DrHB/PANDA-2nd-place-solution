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
PIL.Image.MAX_IMAGE_PIXELS = None
with open('SETTINGS.json', 'r') as fp: settings = json.load(fp)
    
    
FOLD = settings['FOLD']
TRAIN_PREPROC = Path(settings['CLEAN_DATA_DIR'])
CUSTOM_DATA = Path(settings['CUSTOM_DATA'])
N = settings['STAGE_1_N']

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

    
imgs = df['image_id'].to_list()   
fn_dict = {i: [f'{str(TRAIN_PREPROC)}/{i}_{k}.png' for k in range(N)] for i in imgs}
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])



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
    
    
def open_image(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) 
        x = PIL.Image.open(fn).convert(convert_mode)
    if after_open: x = after_open(x)
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(x) 

class MImage(ItemBase):
    def __init__(self, imgs):
        self.obj, self.data = \
          (imgs), [(imgs[i].data - mean[...,None,None])/std[...,None,None] for i in range(len(imgs))]
    
    def apply_tfms(self, tfms,*args, **kwargs):
        for i in range(len(self.obj)):
            self.obj[i] = self.obj[i].apply_tfms(tfms, *args, **kwargs)
            self.data[i] = (self.obj[i].data - mean[...,None,None])/std[...,None,None]
        return self
    
    def __repr__(self): return f'{self.__class__.__name__} {img.shape for img in self.obj}'
    def to_one(self):
        img = torch.stack(self.data,1)
        img = img.view(3,-1,N,224,224).permute(0,1,3,2,4).contiguous().view(3,-1,224*N)
        return Image(mean[...,None,None]+img*std[...,None,None])

class MImageItemList(ImageList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __len__(self)->int: return len(self.items) or 1 
    
    def get(self, i):
        fn = self.items[i]
        fnames = fn_dict[fn.split('.//')[1]]
        imgs = [open_image(fname, convert_mode=self.convert_mode, after_open=self.after_open)
               for fname in fnames]
        return MImage(imgs)

    def reconstruct(self, t):
        return MImage([mean[...,None,None]+_t*std[...,None,None] for _t in t])
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(300,50), **kwargs):
        rows = min(len(xs),8)
        fig, axs = plt.subplots(rows,1,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()
        
tfms = get_transforms(flip_vert=True, max_rotate=10)

optR = OptimizedRounder()
coefficients = [0.5, 1.5, 2.5, 3.5, 4.5]

def qkp_class(y_hat, y):
    y_hat = torch.argmax(F.softmax(y_hat[0]), dim=1)
    return torch.tensor(cohen_kappa_score(y_hat.cpu(), y.cpu(), weights='quadratic'),device='cuda:0')


def qkp_regres(y_hat, y):
    p = optR.predict(y_hat[1].cpu().numpy(), coefficients)
    return torch.tensor(cohen_kappa_score(p, y.cpu(), weights='quadratic'),device='cuda:0')

def qkp_combine(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round((y_hat[1] + torch.argmax(F.softmax(y_hat[0]), dim=1))/2).cpu(), y.cpu(), weights='quadratic'),device='cuda:0')
class CustomLoss(nn.Module):
    def __init__(self, loss_ce, loss_mse):
        super().__init__()
        self.loss_ce  = loss_ce
        self.loss_mse = loss_mse 
        
    def forward(self, i, o):
        loss_mserr = self.loss_mse(i[1], o.float())
        loss_cross = self.loss_ce(i[0], o)
        return loss_cross + loss_mserr
    
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
    def __init__(self):
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
        
       
        
    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
        x = self.enc(x)
        shape = x.shape
        x = x.view(-1, n, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4).contiguous().\
        view(-1, x.shape[1], x.shape[2] * n, x.shape[3])
        x = x.view(x.shape[0], x.shape[1], x.shape[2]//int(np.sqrt(N)), -1)
        x = self.cb(x)
        x = self.head(x)
        return x
    
    
def get_data(df, fold, bs, sz, num_workers):
    return (MImageItemList.from_df(df, path='.', folder='', cols='image_id')
      .split_by_idx(df.index[df.split == fold].tolist())
      .label_from_df(cols=['isup_grade'])
      .transform(tfms=tfms,size=sz,padding_mode='reflection')
      .databunch(bs=bs, num_workers =num_workers))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers", 
        default=32, 
        type=int, 
        help="Number of cpu to use to process data"
    )
    
    parser.add_argument(
        "--bs", 
        default=52, 
        type=int, 
        help="Number of bs"
    )
    
    parser.add_argument(
        "--epoch", 
        default=60, 
        type=int, 
        help="Number of Epochs"
    )
    
    args = parser.parse_args()

    SZ =224
    BS = args.bs
    NAME = 'EXP_80'
    SUFFIX =f'RESNET_34_TILES_{N}_SQ_FT_NBN_SE_DUAL'
    
    data = get_data(df, FOLD, sz=SZ, bs=BS, num_workers=args.num_workers)
    model =Model()
    
    print (f'FOLD:  {FOLD}')
    print (F"TILES: {N}")
    print (F"BS:    {BS}")
    print (F"EPOCH: {args.epoch}")


    learn = Learner(data, model, 
                    metrics=[qkp_class, qkp_regres, qkp_combine])
    learn.split([model.head])
    learn.loss_func = CustomLoss(nn.CrossEntropyLoss(), nn.MSELoss())
    learn.model =  nn.DataParallel(learn.model)
    learn.to_fp16()
    learn.unfreeze()

    learn.fit_one_cycle(args.epoch, max_lr=1e-3, 
                            callbacks = [SaveModelCallback(learn, every='epoch', monitor='valid_loss', name = f'{NAME}_{SUFFIX}_{FOLD}_sq_features'),  
                                         CSVLogger(learn, filename = f"{settings['LOGS_DIR']}/{NAME}_{SUFFIX}_{FOLD}_sq_features.csv")])
    
if __name__=='__main__':
    main()
 

    