import json
settings={
    "RAW_DATA_DIR":'../../prostate-cancer-grade-assessment/',
    "PREDICTION_DIR":"../../prediction/",
    "LEVEL1_IMAGE_DIR":'../data/level1/image/',
    "LEVEL1_MASK_DIR":'../data/level1/mask/',
    "LEVEL0_TILE_DIR":"../data/highresolution/",
    "LEVEL0_JSON_DIR":"./meta_json/",
    "PRED_WEIGHTS_RGUO":"./pred_weights/",
    "LEVEL1_WEIGHTS_RGUO":"./result_level1/",
    "LEVELHALF_WEIGHTS_RGUO":"./result_levelhalf/",
    "CSV_PATH":"./tools/data/pred_v3.csv",
    "SUSPICIOUS_PATH":"tools/data/PANDA_Suspicious_Slides.csv"
}

with open("SETTINGS.json","w+") as f:
    json.dump(settings,f)