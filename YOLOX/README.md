# YOLOX

YOLOX is an object detection model derived from the repository: [Megvii-BaseDetection](https://github.com/Megvii-BaseDetection/YOLOX)

## Run Locally

Clone the project:

```bash
  git clone https://github.com/hamilamailee/Expressive.git
```

Navigate to the project directory:

```bash
  cd Expressive/YOLOX
```

Create a conda environment from the `yolo.yml` file and activate it:

```bash
  conda env create -f yolo.yml
  source activate yolo # or conda activate yolo
```
## Dataset

## Training

To use YOLOX on our dataset, we have selected two versions: YOLOX-Tiny and YOLOX-l. The training of both models follows the default settings provided by the authors in the original repository, and the files for downloading the initial pretrained weights can be found at the [YOLOX-l](https://drive.google.com/file/d/13ZChAp4VTmE5L-0NLEaibag98gurBR0s/view?usp=sharing) and [YOLOX-Tiny](https://drive.google.com/file/d/1kSIWV-CEEMtdHgs0grh_qw7m0eziqlVi/view?usp=drive_link) links, respectivley. After downloading the `.pth` files from the provided links, place them in the `Expressive/YOLOX/` directory. 

```bash
  # YOLOX-l Training
  python tools/train.py -f exps/ExpConfigs/yolox_voc_l.py -d 1 -b 8 --fp16 -o -c yolox_l.pth
```
```bash
  # YOLOX-Tiny Training
  python tools/train.py -f exps/ExpConfigs/yolox_voc_t.py -d 1 -b 8 --fp16 -o -c yolox_tiny.pth
```
After training, the best checkpoints will be saved in the `YOLOX_outputs/yolox_voc_l` and `YOLOX_outputs/yolox_voc_t` directories, under the name `best_ckpt.pth`. 

> If you want to use the pretrained weights, you can download them from this [link](https://drive.google.com/drive/folders/1wdWdzIgH2G84_RILtLOEddCRh8MBC-dB?usp=drive_link), which follows the same structure as `YOLOX_outputs`.

## Evaluating

Since YOLOX and Faster R-CNN use different methods to calculate Average Precision and IoU metrics, the best approach is to export COCO annotations during evaluation and compare all results against ground-truth bounding boxes. To achieve this, the code has been modified to export COCO annotations of predicted boxes to the `dataset` directory by default. Instructions on the contents and arrangement of the `dataset` folder are available [here](##Dataset).

```bash
  # YOLOX-l Evaluating
  python tools/eval.py -n yolox-l -f exps/ExpConfigs/yolox_voc_l.py -d 1 -b 8 --fp16 -c ./YOLOX_outputs/yolox_voc_l/best_ckpt.pth --conf 0.001
```
```bash
  # YOLOX-Tiny Evaluating
  python tools/eval.py -n yolox-t -f exps/ExpConfigs/yolox_voc_t.py -d 1 -b 8 --fp16-c ./YOLOX_outputs/yolox_voc_t/best_ckpt.pth --conf 0.001
```
