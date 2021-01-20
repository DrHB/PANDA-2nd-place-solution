import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from dataset.patch_extraction import PANDAPatchExtraction
import json
torch.backends.cudnn.benchmark = False

def main():
    tiff_dir=SETTINGS['RAW_DATA_DIR']+"train_images/"
    mask_dir=SETTINGS['RAW_DATA_DIR']+"train_label_masks/"

    df=pd.read_csv(SETTINGS["CSV_PATH"])
    suspicious = pd.read_csv(SETTINGS["SUSPICIOUS_PATH"]).image_id
    df = df[~df.image_id.isin(suspicious)]

    dataset=PANDAPatchExtraction(df,
                                 tiff_dir,
                                 mask_dir,
                                 # Patch parameter
                                 patch_size=192,
                                 bg_threshold=0.95,
                                 trail_offset=[0,1/2],
                                 # Augmentation & Normalization
                                 mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225),
                                 )

    dataloader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=8)
    result_metadata=dict()
    for images,coords,img_id,patch_labels,ignore_mask,best_offset in tqdm(dataloader):
        img_id=img_id[0]
        result_metadata[img_id] = best_offset.item()
    #print(result_metadata)
    with open(SETTINGS["LEVEL0_JSON_DIR"]+"Validation_mode.json","w+") as f:
        json.dump(result_metadata,f)

import json
with open("./SETTINGS.json") as f:
    SETTINGS=json.load(f)


if __name__=="__main__":
    main()