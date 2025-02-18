import os
import shutil

import pandas as pd
import fiftyone as fo

from PIL import Image
from tqdm import tqdm

def export_annotation(root_file, images_path, exp_name):
    """_summary_

    Args:
        root_file (str): A path to the results of the current experiment
        images_path (str): A path to the folder containing the images for the annotations
        exp_name (str): A name to define the final json file being written
    """
    df = pd.DataFrame([])

    for i in os.listdir(root_file):
        if not i.endswith(".txt"):
            continue
        new_df = pd.read_csv(os.path.join(root_file, i), header=None, sep=' ')
        if "eye" in i:
            new_df['target'] = 0
        elif "mouth" in i:
            new_df['target'] = 1
        df = pd.concat([df, new_df], ignore_index=True, axis=0)

    df['x_centre'] = (df[2] + df[4]) / 2
    df['y_centre'] = (df[3] + df[5]) / 2
    df['width'] = df[4] - df[2]
    df['height'] = df[5] - df[3]

    images = df[0].unique().tolist()
    image_sizes = pd.DataFrame(images)

    for img in images:
        pil_image = Image.open(os.path.join(images_path, img+".jpg"))
        image_sizes.loc[image_sizes[0] == img, 'img_width'] = pil_image.size[0]
        image_sizes.loc[image_sizes[0] == img, 'img_height'] = pil_image.size[1]

    full_df = pd.merge(df, image_sizes, on=0, how='left')

    full_df['img_name_f'] = full_df[0]
    full_df['target_f'] = full_df['target']
    full_df['x_centre_f'] = full_df['x_centre'] / full_df['img_width']
    full_df['y_centre_f'] = full_df['y_centre'] / full_df['img_height']
    full_df['width_f'] = full_df['width'] / full_df['img_width']
    full_df['height_f'] = full_df['height'] / full_df['img_height']
    full_df['confidence_f'] = full_df[1]

    columns = [col for col in full_df.columns if str(col).endswith("_f")]
    full_df = full_df[columns]

    shutil.rmtree(os.path.join(root_file, "labels"), ignore_errors=True)
    os.mkdir(os.path.join(root_file, "labels"))
        
    print("Writing label annotations...")
    for index, rows in tqdm(full_df.iterrows()):
        with open(os.path.join(root_file, "labels", rows['img_name_f']+".txt"), 'a') as f:
            f.writelines([' '.join([str(i) for i in rows.to_list()[1:]]), '\n'])
            f.close()
        
    if "dataset" in fo.list_datasets():
        print("Deleting existing datasets...")
        dataset = fo.load_dataset("dataset")
        dataset.delete()
        
    name = "dataset"
    classes = ['eye', 'mouth']
    # Import dataset by explicitly providing paths to the source media and labels
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv4Dataset,
        data_path=images_path,
        labels_path=os.path.join(root_file, "labels"),
        classes=classes,
        name=name,
    )

    print(f"Writing COCO annotation format at {os.path.join(root_file, f'{exp_name}_coco_annotations.json')}")
    dataset.export(
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=os.path.join(root_file, f"{exp_name}_coco_annotation.json"),
        label_field="ground_truth",
        abs_paths=False,
    )