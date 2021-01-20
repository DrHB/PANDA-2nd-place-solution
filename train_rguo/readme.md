Hello!

Below you can find a outline of how to reproduce my solution for the Prostate cANcer graDe Assessment (PANDA) Challenge competition.
If you run into any trouble with the setup/code or have any questions please contact me at ruiguo97@126.com

### ARCHIVE CONTENTS
prostate-cancer-grade-assessment   :competition data
data                                                         :cached data for trainning
train_code                                              :code to rebuild models from scratch

### HARDWARE: (The following specs were used to create the original solution)
Ubuntu 16.04 LTS (1024 GB boot disk)
INTEL Xeon® E5-2678 v3  (64GB Memory) 
1 x NVIDIA RTX TITAN 

### SOFTWARE (python packages are detailed separately in `requirements.txt`):
```shell
apt-get update  
apt-get install wget htop nano glib-2.0 unzip git  
pip install opencv-python numpy albumentations tqdm numba pandas sklearn  
git clone https://github.com/NVIDIA/apex  
cd apex  
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
```

### DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)

below are the shell commands used in each step, as run from the top level directory

```shell
kaggle competitions download -c prostate-cancer-grade-assessment  
unzip -q prostate-cancer-grade-assessment.zip -d ./
```

### DATA PROCESSING
1) Prepare Data for level1 (save level1 images in jpeg format)  

```shell
cd train_code   
python crop_level1.py
```

2) Prepare Data for half-level0 (save cropped tiles for half-scale level0 images) 
There are two options 
a) Use trained level1 weights to crop tiles, first need to train 5fold level1 models, and save weights in ./result_level1 (overwrites json file for tiles)

```shell
cd train_code 
sh prepare_level0.sh
```

b) Use my prepared tiles on kaggle. Download these three datasets and unzip them to ../data/highresolution/ 
https://www.kaggle.com/rguo97/panda-cropped-mode0 
https://www.kaggle.com/rguo97/panda-cropped-mode1 
https://www.kaggle.com/rguo97/panda-cropped-mode2 

### MODEL BUILD: There are two options to produce the solution.
1) prediction from trained weights
    a) runs in 3 hour 
    b) uses same weight in best submission
2) retrain models
    a) expect this to run about two weeks
    b) trains all models from scratch

shell command to run each build is below
#1) very fast prediction (create submission.csv) (make sure test_images folder exists)

```shell
cd train_code
python predict_rguo_part1.py
python predict_rguo_part2.py
```

#2) retrain models (overwrites models weights in weights directory and make new cropped tiles for half-scale level0)

```shell
cd train_code 
sh train_all.sh
```


