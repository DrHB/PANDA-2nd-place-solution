import pandas as pd
from sklearn.model_selection import StratifiedKFold


train=pd.read_csv("../prostate-cancer-grade-assessment/train.csv")
#train=train.set_index("image_id")
train['fold']=0
kfold=StratifiedKFold(n_splits=5,random_state=30,shuffle=True)
X=train['image_id']
y=train['isup_grade']

for i,(train_idx,valid_idx) in enumerate(kfold.split(X,y)):
    train.iloc[valid_idx,4]=i

print(train)
train.to_csv("./train_with_fold.csv",index=False)