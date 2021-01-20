Hello!

Below you can find a outline of how to reproduce Cateek's part of the "Save the prostate" team's solution for the Prostate cANcer graDe Assessment (PANDA) Challenge competition.

If you run into any trouble with the setup/code or have any questions please contact me at zubarev.ia@gmail.com


#ARCHIVE CONTENTS
create_images.py   : saving WSI images in png format, initial preprocessing
models.py          : model description and helper functions
dataloader.py      : image processing, augmentations and dataloader with helpers 
main.py            : main training/validation script
utils.py           : auxillary functions


saved image is provided for reference


#HARDWARE: (The following specs were used to create the original solution)
Ubuntu 18.04 LTS 
4 x NVidia 1080TI GPU
AMD Threadripper 1900x CPU


#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.5.1
CUDA 10.2
cuddn 7.6.0.5
nvidia drivers v.440


#DATA PROCESSING
# training/prediction require preprocessing WSI images using create_images.py script.


#MODEL TRAINING
# launch main.py, providing necessary parameters in the header or via docopt. Default parameters are set.


#PREDICTION:
# generating submission.csv is done via team's Kaggle kernel or via predict.py
