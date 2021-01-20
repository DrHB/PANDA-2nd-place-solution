echo "Training level1 models"
python train_msesmooth_se.py 0
python train_msesmooth_se.py 1
python train_msesmooth_se.py 2
python train_msesmooth_se.py 3
python train_msesmooth_se.py 4
echo "Copying level1 5-fold weights to pred_weights"
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold0/se_resnext50_32x4d_fold0_bestOptimQWK.pth ./result_level1/se_resnext50_32x4d_fold0_bestOptimQWK.pth
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold1/se_resnext50_32x4d_fold1_bestOptimQWK.pth ./result_level1/se_resnext50_32x4d_fold1_bestOptimQWK.pth
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold2/se_resnext50_32x4d_fold2_bestOptimQWK.pth ./result_level1/se_resnext50_32x4d_fold2_bestOptimQWK.pth
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold3/se_resnext50_32x4d_fold3_bestOptimQWK.pth ./result_level1/se_resnext50_32x4d_fold3_bestOptimQWK.pth
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold4/se_resnext50_32x4d_fold4_bestOptimQWK.pth ./result_level1/se_resnext50_32x4d_fold4_bestOptimQWK.pth

cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold0/se_resnext50_32x4d_fold0_bestOptimQWK.pth ./pred_weights/se_resnext50_32x4d_fold0_bestOptimQWK.pth
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold1/se_resnext50_32x4d_fold1_bestOptimQWK.pth ./pred_weights/se_resnext50_32x4d_fold1_bestOptimQWK.pth
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold2/se_resnext50_32x4d_fold2_bestOptimQWK.pth ./pred_weights/se_resnext50_32x4d_fold2_bestOptimQWK.pth
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold3/se_resnext50_32x4d_fold3_bestOptimQWK.pth ./pred_weights/se_resnext50_32x4d_fold3_bestOptimQWK.pth
cp ./result_se/se_resnext50_32x4d_regression_level1_msesmooth_fold4/se_resnext50_32x4d_fold4_bestOptimQWK.pth ./pred_weights/se_resnext50_32x4d_fold4_bestOptimQWK.pth

echo "Prepare level0 tiles"
sh prepare_level0.sh

echo "Training level1 efb0"
python train_msehuber_efb0.py 0
python train_msehuber_efb0.py 1
python train_msehuber_efb0.py 2
python train_msehuber_efb0.py 3
python train_msehuber_efb0.py 4

cp ./result_efb0/efficientnet-b0_regression_level1_msehuber_fold0/efficientnet-b0_fold0_bestOptimQWK.pth ./result_level1/efficientnet-b0_fold0_bestOptimQWK.pth
cp ./result_efb0/efficientnet-b0_regression_level1_msehuber_fold1/efficientnet-b0_fold1_bestOptimQWK.pth ./result_level1/efficientnet-b0_fold1_bestOptimQWK.pth
cp ./result_efb0/efficientnet-b0_regression_level1_msehuber_fold2/efficientnet-b0_fold2_bestOptimQWK.pth ./result_level1/efficientnet-b0_fold2_bestOptimQWK.pth
cp ./result_efb0/efficientnet-b0_regression_level1_msehuber_fold3/efficientnet-b0_fold3_bestOptimQWK.pth ./result_level1/efficientnet-b0_fold3_bestOptimQWK.pth
cp ./result_efb0/efficientnet-b0_regression_level1_msehuber_fold4/efficientnet-b0_fold4_bestOptimQWK.pth ./result_level1/efficientnet-b0_fold4_bestOptimQWK.pth

echo "Training level0 models"
python train_high_efb0.py 0
python train_high_efb0.py 1
python train_high_efb0.py 2
python train_high_efb0.py 3
python train_high_efb0.py 4

echo "Copying level0 5-fold weights to pred_weights"
cp ./result_halflevel/efficientnet-b0_regression_levelhalf_attention_smooth_cosine_largebs_fold0/efficientnet-b0_fold0_bestOptimQWK.pth ./pred_weights/efficientnet-b0_fold0_bestOptimQWK.pth
cp ./result_halflevel/efficientnet-b0_regression_levelhalf_attention_smooth_cosine_largebs_fold1/efficientnet-b0_fold1_bestOptimQWK.pth ./pred_weights/efficientnet-b0_fold1_bestOptimQWK.pth
cp ./result_halflevel/efficientnet-b0_regression_levelhalf_attention_smooth_cosine_largebs_fold2/efficientnet-b0_fold2_bestOptimQWK.pth ./pred_weights/efficientnet-b0_fold2_bestOptimQWK.pth
cp ./result_halflevel/efficientnet-b0_regression_levelhalf_attention_smooth_cosine_largebs_fold3/efficientnet-b0_fold3_bestOptimQWK.pth ./pred_weights/efficientnet-b0_fold3_bestOptimQWK.pth
cp ./result_halflevel/efficientnet-b0_regression_levelhalf_attention_smooth_cosine_largebs_fold4/efficientnet-b0_fold4_bestOptimQWK.pth ./pred_weights/efficientnet-b0_fold4_bestOptimQWK.pth

echo "Finished"