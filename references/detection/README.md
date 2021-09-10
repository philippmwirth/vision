# Lightly Active Learning Benchmarks Object Detection

This folder contains the torchvision reference training scripts but adapted to benchmark lightly active learning.

The purpose of the adaptations was to 
1. Show how easy it is to integrate active learning with Lightly into existing training scripts.
2. Benchmark the different active learning samplers offered by Lightly.


## Setup

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

In addition to the torchvision setup, `lightly>=1.1.4` is required to run the benchmarks:
```
pip install lightly
```


## Benchmarks


### Faster R-CNN ResNet-50 FPN VOC 2012
```
rm outputs/VOCDetection.json # remove outputs if you're starting a new run
python voc_al_benchmark.py --token [YOUR_TOKEN] --data-path [YOUR_DATA_PATH] --steps 1143 2286 3429 4572 5717 --lr 0.003 --pretrained
python plot_a_log.py outputs/VOCDetection.json
```

### Faster R-CNN ResNet-50 FPN KITTI
Coming soon...




# Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Except otherwise noted, all models have been trained on 8x V100 GPUs. 

### Faster R-CNN ResNet-50 FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Faster R-CNN MobileNetV3-Large FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Faster R-CNN MobileNetV3-Large 320 FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_320_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### RetinaNet
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01
```

### SSD300 VGG16
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model ssd300_vgg16 --epochs 120\
    --lr-steps 80 110 --aspect-ratio-group-factor 3 --lr 0.002 --batch-size 4\
    --weight-decay 0.0005 --data-augmentation ssd
```

### SSDlite320 MobileNetV3-Large
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model ssdlite320_mobilenet_v3_large --epochs 660\
    --aspect-ratio-group-factor 3 --lr-scheduler cosineannealinglr --lr 0.15 --batch-size 24\
    --weight-decay 0.00004 --data-augmentation ssdlite
```


### Mask R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Keypoint R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3
```

