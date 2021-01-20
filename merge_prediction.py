import pandas as pd
import numpy as np
def predict(X, coef=[0.5,1.5,2.5,3.5,4.5]):
    return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3,4,5])
res_med_se50=pd.read_csv("prediction/submission_medianse50.csv")['isup_grade']
res_high_efb0=pd.read_csv("prediction/submission_highefb0.csv")['isup_grade']
res_drhb=pd.read_csv("prediction/submission_drhb.csv")['isup_grade']
res_xie=pd.read_csv("prediction/submission_xie29.csv")['isup_grade']
res_igor=pd.read_csv("prediction/submission_cateek.csv")['isup_grade']
image_id=pd.read_csv("prediction/submission_medianse50.csv")['image_id']
final_result=((res_med_se50+res_igor+res_drhb)/3+res_high_efb0+res_xie)/3
final_result=predict(final_result)

result_df=pd.DataFrame({
    "image_id":image_id,
    "isup_grade":isup_grade
})
result_df.to_csv("prediction/submission.csv",index=False)