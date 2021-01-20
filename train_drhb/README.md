Hello!

Below you can find a outline of how to reproduce my solution for the PANDA competition.
If you run into any trouble with the setup/code or have any questions please contact me at habib.s.t.bukhari@gmail.com

### ARCHIVE CONTENTS
train_drhb/models: 	contain original model that used for the best submission

### HARDWARE: (The following specs were used to create the original solution)
Ubuntu
2 X RTX TITAN
32 CPU
128 GB RAM


### SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.7
CUDA 10.1
nvidia drivers 418.74

### DATA SETUP (more deatils in `directory_structure.txt`)

#### drhb/data
assumes your data is inside the folder `drhb/data` and contains following subfolders
<br /> `train_images` - collection of training tiff files
<br /> `test_images`  - collection of test tiff file
<br /> `processed/train.csv`- containing training csv file same as during kaggle competitons except it should contain additional column called `split` which should have valius (`0` will use as validation image, `1` will be used as training example this is only needed if we want to train from scratch
<br /> `processed/test.csv`- csv file with column `image_id`.

#### drhb/data_custom
<br /> `MASTER_PANDAS.csv` - modified version of original `train.csv`, contains additinal column called `split` needed for repoducing submitted model
<br /> `PANDA_Suspicious_Slides.csv` list of bad slides taken from (https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/151323) . This slides are ignored during training phase


### DATA PROCESSING

`python prepare_data.py --num_workers 32` (`note`: num_workers can be reduced to 4, 8, 16)
<br /> 1) Read training data from `RAW_DATA_DIR` (specified in `SETTINGS.json`)
<br /> 2) Save the cleaned data to `CLEAN_DATA_DIR` (specified in `SETTINGS.json`)


### MODEL BUILD: There are three options to produce the solution.
1) getting prediction for new test data
    a) runs in a few minutes
    b) uses stored weights from the best submison.

2) retraining model from scratch
    a) expect this to run 14 hour
    b) trains  models from scratch

### 1) getting prediction for new test data
<br /> `python predict.py` will use best submitted kaggle model (`EXP_80_RESNET_34_TILES_81_SQ_FT_NBN_SE_DUAL_0_sq_features_41_best.pth`) from directory `models`
<br />1) Read csv file `TEST_DATA_CLEAN_PATH` (specified in `SETTINGS.json`) - this `csv` file should contain `image_id` columsn which has id of images similar to `train.csv`
<br />2) Read test data from `RAW_DATA_TEST` (specified in `SETTINGS.json`) - stored raw `tiff` files
<br />3) will generate prediction `submission.csv` in directory `SUBMISSION_DIR` (specified in `SETTINGS.json`)

### 2) retraining model from scratch to reproduce results
Model is trainied in two phases
<br /> 1) Phase 1 - trains model on 49 tiles
<br /> 2) Select best checkpoint by runing TTA
<br /> 2) Phase 2 - uses weight from Phase 1 to finetune moel on 81 tiles

#### phase_1
<br /> `python train_phase_1.py` will use csv file `drhb/data_custom` to read preprocessd images from `CLEAN_DATA_DIR` (specified in `SETTINGS.json`) which were generated in `DATA PROCESSING`. This phase will run for 60 epochs (approx 8 min per epoch). All the checkpoints will be stored in `drhb/models`. All the log will be stored in `LOGS_DIR` (specified in `SETTINGS.json`).

#### Select best checkpoint by runing tta
<br /> 1) After runing `phase 1` please open `LOGS_DIR` (specified in `SETTINGS.json`) and have a look at log file `EXP_80_RESNET_34_TILES_49_SQ_FT_NBN_SE_DUAL_0_sq_features.csv.csv` . Select last few epoch judging by `qkp_combine` and `qkp_regres` (higher the score is better).

<br /> 2) After selecting few or one best epoch run the following command `python select_model.py --phase 1 --epoch 58` . This will run TTA on validation data using checkpoint generated at epoch `58` (`58` here just an example, it can be any epoch from `0` to `59`. This will produce a new csv log file (1 or several depending on how many epoch you decide to select) located in `LOGS_DIR` (specified in `SETTINGS.json`) called `TTA_EXP_80_RESNET_34_TILES_49_SQ_FT_NBN_SE_DUAL_0_sq_features_58.csv` (it will end with the number of `epoch` which you selected in this example I am using epoch `58`). You have to judge score overall and for each datacenter (higher is better). Choose the best checkpoint and use it to run `Phase 2` finetuning

<br /> 3) Phase 2 finetuning (approx 12 hours). Once you selected best checkpoint from Phase 1 run following command. `python train_phase_2.py --phase_1_best_epoch 58`. Here we are finething on `81` tiles using best epoch from phase 1 (in this example its epoch `58`). Once trainin is done open `LOGS_DIR` (specified in `SETTINGS.json`) and have a look at log file `EXP_80_RESNET_34_TILES_81_SQ_FT_NBN_SE_DUAL_0_sq_features.csv.csv` . Select last few epoch judging by `qkp_combine` and `qkp_regres` (higher the score is better). Run `python select_model.py --phase 2 --epoch 45` This will run TTA on validation data using checkpoint generated at epoch `45` (`45` here just an example, it can be any epoch from `0` to `59`. This will produce a new csv log file (1 or several depending on how many epoch you decide to select). Select model which produces hgiher score and then run full inference on test data as in #1 (`getting prediction for new test data`) but with small modifcation `python predict.py --model_name EXP_80_RESNET_34_TILES_81_SQ_FT_NBN_SE_DUAL_0_sq_features_45.pth`. Here we are using best checpoint from phase 2 to run inference on test data. New submission csv will be generated in subsmisson folder.