
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

## Training

To use YOLOX on our dataset, we have selected two versions: YOLOX-Tiny and YOLOX-l. The training of both models follows the default settings provided by the authors in the original repository, and the files for downloading the initial pretrained weights can be found at the [YOLOX-l](https://drive.google.com/file/d/13ZChAp4VTmE5L-0NLEaibag98gurBR0s/view?usp=sharing) and [YOLOX-Tiny](https://drive.google.com/file/d/1kSIWV-CEEMtdHgs0grh_qw7m0eziqlVi/view?usp=drive_link) links, respectivley. After downloading the `.pth` files from the provided links, place them in the `Expressive/YOLOX/` directory. 

```bash
  # YOLOX-l Training
  python tools/train.py -f exps/ExpConfigs/yolox_voc_l.py -d 1 -b 8 --fp16 -o -c yolox_l.pth

  # YOLOX-Tiny Training
  python tools/train.py -f exps/ExpConfigs/yolox_voc_t.py -d 1 -b 8 --fp16 -o -c yolox_tiny.pth
```
After training, the best checkpoints will be saved in the `YOLOX_outputs/yolox_voc_l` and `YOLOX_outputs/yolox_voc_t` directories, under the name `best_ckpt.pth`. 

> If you want to use the pretrained weights, you can download them from this [link](https://drive.google.com/drive/folders/1wdWdzIgH2G84_RILtLOEddCRh8MBC-dB?usp=drive_link), which follows the same structure as `YOLOX_outputs`.
