All scripts are in `train_code` folder

1. `python crop_level1.py`, which would

   - Read tiff file from RAW_DATA_DIR (specified in SETTINGS.json))

   - Create jpeg format for all slides, the useless white region will be dropped and save at LEVEL1_IMAGE_DIR

   - Create png format for masks, if no mask is provided, an empty mask will be generated. Will save at LEVEL1_MASK_DIR

2. `python crop_next_level_parallel.py <fold> <mode>` , which would

   - Crop half-scale level0 tiles for specified fold and different offset mode in tiling.
   - Load trained level1 modes and calculate out-of-fold attention prediction
   - Save tiles at LEVEL0_TILE_DIR
   - Save attention values in LEVEL0_JSON_DIR

3. `python crop_next_level_parallel_mode2.py <fold>`, which would

   - Crop half-scale level0 tiles for specified fold with tiling method https://www.kaggle.com/akensert/panda-optimized-tiling-tf-data-dataset.
   - Load trained level1 modes and calculate out-of-fold attention prediction
   - Save tiles at LEVEL0_TILE_DIR
   - Save attention values in LEVEL0_JSON_DIR
4. `python get_validation_mode.py` , which would
- Calculate best offset mode(0 or 1) used in validation, save at LEVEL0_JSON_DIR.
5. `python train_msesmooth_se.py <fold>` , which would
   - Train se_resnext50_32x4d in level1 resolution, with smoothed labels from oof prediction
   - Save weights to PRED_WEIGHTS_RGUO
6. `python train_msehuber_efb0.py <fold>` , which would
   - Train efficientnet-b0 in level1 resolution, with MSE loss for Karolinska and Huber Loss for Radboud
   - Save weights to LEVEL1_WEIGHTS_RGUO
   - Will be used in weight initialization in `train_high_efb0.py`
7. `python train_high_efb0.py` , which would
   - Train efficientnet-b0 in half-scale level0 resolution
   - Save weights to PRED_WEIGHTS_RGUO
   - Resume from weights in level1
8. `python predict_rguo_part1.py` , which would
   - Make submission for se_resnext50_32x4d on median resolution
   - Save result at PREDICTION_DIR
9. `python predict_rguo_part2.py` , which would
   - Make submission for efficientnet-b0 on high resolution
   - Save result at PREDICTION_DIR



