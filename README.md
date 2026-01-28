<div align="center">

# Prostate Cancer Gleason Grading

### Deep Learning for Histopathology Image Classification

[![Nature Medicine](https://img.shields.io/badge/Nature%20Medicine-Published-E42D25?style=for-the-badge)](https://www.nature.com/articles/s41591-021-01620-2)
[![Kaggle](https://img.shields.io/badge/Kaggle-2nd%20Place-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/c/prostate-cancer-grade-assessment)

**CNN ensemble for automated Gleason grading of prostate cancer from whole-slide biopsy images**

</div>

---

## Problem

Gleason grading of prostate cancer biopsies is critical for treatment decisions but suffers from significant inter-pathologist variability. The PANDA challenge was the largest histopathology AI competition, with 1,290 teams from 65 countries developing algorithms on 10,616 digitized prostate biopsies.

This 2nd place solution uses an ensemble of CNNs (EfficientNet, SE-ResNeXt) with tile-based attention pooling, achieving pathologist-level agreement (quadratic weighted kappa >0.9). This work contributed to the published paper ["Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge"](https://www.nature.com/articles/s41591-021-01620-2) in Nature Medicine.

## Approach

- **Tile Extraction**: Whole-slide images processed as grids of tiles at multiple resolutions
- **Backbone Models**: EfficientNet-B0/B1, SE-ResNeXt50
- **Attention Pooling**: Learnable aggregation of tile-level features to slide-level prediction
- **Two-Phase Training**: Initial training followed by fine-tuning with hard examples
- **Ensemble**: Predictions merged from 4 team members' models

## Repository Structure

| Directory | Author | Description |
|-----------|--------|-------------|
| `train_drhb` | DrHB | 2-phase training with EfficientNet |
| `train_rguo` | R Guo | Attention-based models |
| `train_cateek` | CatEek | SE-ResNeXt models |
| `train_xie29` | xie29 | TFRecords pipeline |

## Links

- [Competition Page](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
- [Solution Discussion](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169108)
- [Nature Medicine Paper](https://www.nature.com/articles/s41591-021-01620-2)

## Keywords

`prostate-cancer` `gleason-grading` `histopathology` `digital-pathology` `deep-learning` `pytorch` `whole-slide-images` `medical-imaging` `kaggle-competition`

---

## Reproducing Results

### Archive Contents
```
prostate-cancer-grade-assessment    :competition data
train_cateek                        :train code from CatEek
train_drhb                          :train code from DrHB
train_rguo                          :train code from R Guo
train_xie29                         :train code from xie29
prediction                          :prediction files
```
### Train Details

For more train details, please refer to each train folder.

### Prediction

To get prediction, prepare the environment and run:

```shell
sh make_pred.sh
```

---

<div align="center">

**Star this repo if you find it useful!**

</div>

