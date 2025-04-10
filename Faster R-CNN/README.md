# Faster R-CNN

The Faster R-CNN object detection model has beeen derived from the repository: [here](https://github.com/pytorch/vision/tree/main/references/detection)

## Run Locally

Clone the project:

```bash
  git clone https://github.com/hamilamailee/Expressive.git
```

Navigate to the project directory:

```bash
  cd "Expressive/Faster R-CNN"
```

Create a conda environment from the `fasterrcnn.yml` file and activate it:

```bash
  conda env create -f fasterrcnn.yml
  source activate fasterrcnn # or conda activate fasterrcnn
```

## Datasets


## Training

To match the training script (`train.py`) with our dataset, the following adjustments have to be made:

### `coco_eval.py`

```python
class CocoEvaluator:
  ...
  def update(self, predictions):
    ...
    for iou_type in self.iou_types:
      ...
+     try:
+       with open(f'export_ann_results.json', 'a') as f:
+         for ann in coco_dt.dataset["annotations"]:
+           json.dump(ann, f)
+           json.dump(",", f)
+           f.close()
+     except:
+       print("Error with img {} in annotation".format(img_ids))
+     try:
+       with open(f'export_img_results.json', 'a') as f:
+         for img in coco_dt.dataset["images"]:
+           json.dump(img, f)
+           json.dump(",", f)
+           f.close()
+     except:
+       print("Error with img {} in image".format(img_ids))
```
This piece of code saves the evaluation results in COCO format to a JSON file for better comparison and visualization in the next steps.

### `coco_utils.py`
```python
...
class ConvertCocoPolysToMask:
  def __call__(self, image, target):
    ...
#   segmentations = [obj["segmentation"] for obj in anno]
#   masks = convert_coco_poly_to_mask(segmentations, h, w)
    ...
    classes = classes[keep]
#   masks = masks[keep]
    ...
    target["labels"] = classes
#   target["masks"] = masks
    ...
```
Since the annotated data does not contain any `segmentations` or `masks`, those parts have been commented out.
```python
...
def get_coco(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
+       "test": ("test2017", os.path.join("annotations", anno_file_template.format(mode, "test"))),
+       "my_test": ("my_test", os.path.join("annotations", anno_file_template.format(mode, "my_test"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }
    ...
```
To include our train, validation, and test datasets, along with our out-of-domain test cases, we add the lines above to the code to define the paths to our folders. The `test` folder contains the test set used during training, while the `my_test` folder contains out-of-domain instances—specifically, some photos from the `iCartoonFace` dataset and images from _Monsters, Inc._.

### `train.py`
```python
...
def get_dataset(is_train, args):
+   image_set = "train" if is_train else "test"
+   num_classes, mode = {"coco": (3, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    ...
```
The change in `image_set` is important for the different stages of training, validation, and testing. For the testing stage, the value is set to `test`, while `my_test` is used for the out-of-domain instances.

```python
...
def main(args):
    ...
    if args.resume:
+       checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        ...
```
As mentiond in PyTorch documention outlined [here](), we have:
> If loading an old checkpoint that contains an `nn.Module`, we recommend `weights_only=False`. 
