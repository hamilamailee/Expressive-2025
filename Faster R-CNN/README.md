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

To match the training script (`train.py`) with our dataset, the following adjustments have to be made:

### `coco_eval.py`

```python
class CocoEvaluator:
  ...
  def update(self, predictions):
    ...
    with redirect_stdout(io.StringIO()):
      coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
+   try:
+     with open('export_my_test.json', 'a') as f:
+       for ann in coco_dt.dataset["annotations"]:
+         json.dump(ann, f)
+         json.dump(",", f)
+         f.close()
+   except:
+     print("Error with img {}".format(img_ids))
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
