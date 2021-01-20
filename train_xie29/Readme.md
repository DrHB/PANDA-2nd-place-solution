Hello!

Below you can find a outline of how to reproduce my solution for the PANDA competition.
If you run into any trouble with the setup/code or have any questions please contact me at
rayxie0329@gmail.com

### ARCHIVE CONTENTS
xie29/data/panda-best-weights: 	contain original model that used for the best submission

### HARDWARE: (The following specs were used to create the original solution)
Kaggle TPU

### SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.7
<br /> Tensorflow 2.2.0

### DATA SETUP (more deatils in `directory_structure.txt`)

#### xie29/data
<br /> `prostate-cancer-grade-assessment/train_images` - collection of original training data from host/kaggle
<br /> `prostate-cancer-grade-assessment/test_images`  - collection of original testing data from host/kaggle
<br /> `prostate-cancer-grade-assessment/train.csv`- original training data csv file
<br /> `prostate-cancer-grade-assessment/test.csv`- original testing data csv file
<br /> `prostate-cancer-grade-assessment/sample_submission.csv`- original sample submission csv file
<br /> `customize/PANDA_Suspicious_Slides.csv` - list of bad slides taken from (https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/151323) . This slides are ignored during training phase
<br /> `customize/cleanlabv2.csv` - A csv file from my teammate `R GUO`. Except the original information from `train.csv` provided by host, there are also oof predictions from his previous 5 fold models, noisy level from clearnLab, amount of good tiles in each tiff file and data split folding. Only data split folding was used in my solution.
<br /> `tiles_tfrec`- Folder for saving tfrecord files for TPU training, data will be saved in their corresponding folding folder.
<br /> `gcs_path/gcs_paths.csv` - GCS path of tfrecord files from `CLEAN_DATA_DIR`
<br /> `training_checkpoints` - A folder for saving checkpoints of training model
<br /> `panda-best-weights` - Checkpoints for my best model in competition

### DATA PROCESSING

`python prepare_data.py`
<br /> (note: This command will generate tfrecord files from original tiff data, but since I was using Kaggle TPU for model training, so the tfrecord files need to be uploaded to Kaggle server)
<br /> 1) Read training data from `RAW_DATA_DIR` (specified in `SETTINGS.json`)	
<br /> 2) Save the cleaned data to `CLEAN_DATA_DIR` (specified in `SETTINGS.json`)
<br /> 3) Upload the `tpu_prepare_data.ipynb` to kaggle environment notebook
<br /> 4) Upload each fold of tfrecord files in each fold folder at `CLEAN_DATA_DIR` to the same notebook you upload the `tpu_prepare_data.ipynb`, each fold of uploaded dataset should has same name as `TRAIN_DATA_CLEAN_PATH_1~5` in SETTINGS.json. Then upload the SETTINGS.json to the same notebook, the kaggle dataset name of this SETTINGS.json should be `settings_json` or you need to modify the path in `tpu_prepare_data.ipynb`
<br /> 5) Commit the notebook and it will generate csv file which contain gcs_path of these tfrecord files
<br /> 6) Save the generated csv file from that notebook and save it to `data/gcs_path`. This csv contain the gcs path of tfrecord files.

### MODEL BUILD: There are three options to produce the solution.
1) getting prediction for new test data (note : Can run on CPU/GPU/TPU, default setting in SETTINGS.json is on local PC)
    <br /> a) runs in a few minutes
    <br /> b) uses stored weights from the best submison.  

2) retraining model from scratch (note : Only can run on TPU, default setting in SETTINGS.json is on Kaggle TPU)
    a) it will take 6 hours to finish the trainin on 8 cores TPU
    b) trains models from scratch

### 1) getting prediction for new test data
<br /> `python predict.py` will use best submitted kaggle model (`fold0_b3_144tiles_128tilesize_mse_huber_0905kcv_0853rcv_0706.h5` and `fold2_b3_144tiles_128tilesize_mse_huber_09174kcv_0841rcv_0719.h5`) from directory `xie29/data/panda-best-weights` to generate my part of submission file and raw predictions of my part for team ensemble.
<br />1) Read csv file `RAW_DATA_TEST` (specified in `SETTINGS.json`) - this `csv` file should contain `image_id` columsn which has id of images similar to `train.csv`
<br />2) Read test data from `RAW_DATA_TEST` (specified in `SETTINGS.json`), and do the data processing and model prediction in realtime.
<br />3) will generate prediction `submission.csv` in directory `SUBMISSION_DIR` (specified in `SETTINGS.json`) - use to submit
<br />4) will generate raw prediction `xie29_ensemble_submission.csv` in directory `SUBMISSION_DIR` (specified in `SETTINGS.json`) - use to ensemble

### 2) retraining model from scratch to reproduce results (note : Since this script need to run on Kaggle TPU environment, and the time limit of kaggle tpu is 3 hrs each time, so this script need to run 2 times for phase 1 and phase 2 training)

<br /> 1) Phase 1 start
<br /> 1) Upload `train.py` to Kaggle environment notebook, make sure the `TRAIN_EPOCHS` is 25 and `START_EPOCH` is 0 in phase 1
<br /> 2) Add the `SETTINGS.json` which you uploaded at step 4 in data processing to this notebook
<br /> 3) Upload `gcs_paths.csv` to this notebook, and you should name the this kaggle dataset as `gcs_path`
<br /> 4) The `FOLD` defined in `SETTINGS.json` will decide which fold this notebook are going to train. Or you can define it in notebook directly by modifying `TRAIN_FOLD`
<br /> 5) Run the script and the checkpoints will be saved in `MODEL_TRAIN_CHECKPOINT_DIR` (specified in `SETTINGS.json`)
<br /> 6) Save the output checkpoints as kaggle dataset
<br /> 7) Phase 2 start
<br /> 8) Add the kaggle dataset you just saved in this notebook and set the path of `model_finalcheckpoint.h5` to `PRETRAIN_PATH` in script.
<br /> 9) Set the `TRAIN_EPOCHS` to 15 and `START_EPOCH` to 25
<br /> 10) Run the script for phase 2 trainin
<br /> 11) Save the checkpoint of phase 2 which named `kar_model.h5` for your afterward inference
