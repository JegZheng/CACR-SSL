## PyTorch implementation of large-scale label-imbalanced experiments presented in our paper: Contrastive Attraction and Contrastive Repulsion for Representation Learning

### 1.Requirements:
- pytorch==1.6.0;
- tensorboard==2.3.0;
- detectron2==0.3;

### 2. Sturcture of code
ImageNet-1K, Webvisionv1 and Imagenet-22K experiments are in ``examples''.

### 3. Example Usage
(1) Run the codes on ImageNet-1K/Webvisionv1/Imagenet-22K 

(1.1) prepare the dataset 

The file structure should look like:
  ```bash
  $ tree data
  dataset
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```


(1.2) running experiments on ImageNet-1K/Webvisionv1/Imagenet-22K:
Training: 
```python main_ours.py \
                 -a resnet50 \
                 --lr 0.03 \
                 --Ny 4 \
                 --Ns 256 \
                 --gpus 0 1 2 3 4 5 6 7 \
                 --moco-k 65536 \
                 --tau_pos 1.0 \
                 --tau_neg 2.0 \
                 --alpha 1.0 --beta 1.0 \
                 --mlp --aug-plus --cos \
                 --dist-url tcp://localhost:10001 \
                 --multiprocessing-distributed \
                 --world-size 1 \
                 --rank 0 
                 data/imagenet1k (or webvision/imagenet22k)
```

Testing: 
```
python main_lincls.py \
                 -a resnet50 \
                 --lr 30.0 \
                 --batch-size 256 \
                 --gpus 0 1 2 3 4 5 6 7 \
                 --pretrained [encoder_pth] \
                 --dist-url tcp://localhost:10001 \
                 --multiprocessing-distributed \
                 --world-size 1 \
                 --rank 0 
                 data/imagenet1k
```
(webvisionv1 code is provided in the subfolder "webvision", for ImageNet-22K, you only need to change the data path to the corresponding data folder.)

(2) Detection and Segemntation: After you trained the encoder on ImageNet, you can follow the usage in [MoCo](https://github.com/facebookresearch/moco) to conduct the detection and segmentation task.

### Acknowledgements
This code benefits a lot from [MoCo](https://github.com/facebookresearch/moco), we would like to thank them here. 



