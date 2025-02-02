import torch
import torchvision

from os.path import join
from dataset import Manga109COCO
from references.detection import utils, transforms as T
from references.detection.engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

torch.cuda.empty_cache()

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def data_loader_from_split(root, split):
    
    is_train = True if split == 'train' else False
    
    dataset = Manga109COCO(join(root, split, 'data'),
                           join(root, split, 'labels.json'),
                           get_transform(is_train))
    
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=8, 
                                       shuffle=True, 
                                       num_workers=2,
                                       collate_fn=utils.collate_fn)

coco_root = "./manga109_COCO"
splits = ['train', 'test', 'val']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 3

data_loader_train, data_loader_test, data_loader_val = [
    data_loader_from_split(coco_root, split) for split in splits
]

# get the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 3  # 2 class (eye + mouth) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=5e-4)
# optimizer = torch.optim.SGD(params, lr=1e-4,
#                             momentum=0.9, weight_decay=1e-5)
# # learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=3,
#                                                 gamma=0.1)
num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_val, device=device)  # validation evaluation

# evaluate(model, data_loader_test, device=device)
# print("Training, validation, and testing complete!")
