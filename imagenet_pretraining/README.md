## PyTorch implementation of Contrastive Attraction and Contrastive Repulsion for Representation Learning - ImageNet pretraining


### 1.Requirements:


The code has been tested with CUDA 11.0/CuDNN 8, PyTorch 2.0 and timm 0.4.9.

[timm](https://github.com/rwightman/pytorch-image-models) (`timm==0.4.9`) is specified for the training with ViT models.


### 2. Prepare dataset
Download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).
The file structure should look like:
  ```bash
  $ tree data
  imagenet
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


### Usage: Self-supervised Pre-Training

Below are three examples for MoCo v3 pre-training. 

#### ResNet-50 training with batch 4096

On the first node, run:
```
python main_pretrain.py \
  --moco-m-cos --crop-min=.2 \
  --lr=.3 --wd=1.5e-6 --moco-m=0.996 --epochs=1000 \
  --dist-url='tcp://[master node address]:[master node port]' \
  --multiprocessing-distributed --world-size=[num node] --rank=[node id] \ 
  --output_dir=[your experiments folder]
  [your imagenet-folder with train and val folders]
```
We follow [MoCov3](https://github.com/facebookresearch/moco-v3), the pretraining is done with a batch size of 4096, on 2 nodes with a total of 16 Volta 32G GPUs. 


#### ViT-Base with 8-node training, batch 4096

With a batch size of 4096, ViT-Base is trained with 8 nodes:
```
python main_pretrain.py \
  -a vit_base \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --moco-m-cos --moco-t=.2 \
  --multiprocessing-distributed --world-size=[num node] --rank=[node id] \ 
  --output_dir=[your experiments folder]
  [your imagenet-folder with train and val folders]
```
Here, we use 4-8xA100, i.e., 32 A100 in total for the experiments. 

#### Notes:
1. We use RBF kernel cost by default. You may also swtich to Euclidean cost by setting `--cacr-cost=euclidean`.
2. To enable multiple local crops proposed in [Swav](), please turn on with `--use-local-crops`. 
3. In the multi-crop setting, we by default use a light mode, which already provide satisfactory performance reported in the paper. To boost better performance, we may turn on the full mode with `--use-full-mode`.
4. If the experiment is conducted on single node, we may simply set `--dist-url='tcp://localhost:10001'`.
5. Other configuration details are the same as [MoCov3](https://github.com/facebookresearch/moco-v3).

### Usage: k-NN

We provide k-NN script for fast evaluation. Simply run:

```
python -m torch.distributed.launch --nproc_per_node=[num GPU] 
main_knn.py \
    --arch ${arch} \
    --pretrained_weights [your ckpt path] \
    --data_path [your imagenet-folder with train and val folders] \
    --dump_features [your experiments folder for feature storage] /
```

### Usage: Linear Classification


```
python main_lincls.py \
  -a [architecture] --lr [learning rate] \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your ckpt path] \
  --output_dir [your experiments folder] \
  [your imagenet-folder with train and val folders]
```

### Main results

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">model</th>
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">linear<br/>acc</th>
<th valign="center">k-NN<br/>acc</th>
<th valign="center">checkpoint</th>
<!-- TABLE BODY -->
<tr>
<td align="left">ResNet50</td>
<td align="right">1000</td>
<td align="center">74.7</td>
<td align="center">69.4</td>
<td align="center"><a href="https://drive.google.com/file/d/17mE7obaWAG0-YMu3ffZLK-DkhNmn2hhB/view?usp=sharing">download</a></td>
</tr>
<tr>
<td align="left">ViT-Base</td>
<td align="right">300</td>
<td align="center">77.1</td>
<td align="center">72.6</td>
<td align="center"><a href="https://drive.google.com/file/d/19BRxMvhdYcCLFHfgGQ-FZQibAN9IdurK/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

### Acknowledgements
This code benefits a lot from [MoCov3](https://github.com/facebookresearch/moco-v3), we would like to thank them here. 
