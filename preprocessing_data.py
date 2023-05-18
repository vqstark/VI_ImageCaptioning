from config import Config
import json


C = Config()

# Read data
def load_data():
    with open(C.train_data_path, encoding="utf8") as f:
        train_data = f.read()

    train_data = json.loads(train_data)
    f.close()
    print('Read train data: OK')

    with open(C.val_data_path, encoding="utf8") as f:
        val_data = f.read()

    val_data = json.loads(val_data)
    f.close()
    print('Read val data: OK')

    return train_data, val_data


# Remove unnecessary infomation from dataset
# Stored all annotations by list
# Add <start>, <end> tag for each sequence
def cleaning_data(json):
    imgs_list = json['images']
    annotations_list = json['annotations']
    data = []
    
    for img in imgs_list:
        img_id = img['id']
        # Get all annotations
        annos_of_img = []
        for anno in annotations_list:
            if anno['image_id']==img_id:
                annos_of_img.append('<start> '+anno['caption']+' <end>')
        img_annos = {}
        img_annos['url'] = img['coco_url']
        img_annos['annotations'] = annos_of_img
        data.append(img_annos)
    return data