# Code For CHAIN+FastGANDBig

### Download [Inception model](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) to path
```
data/inception_model
```
The inception model is converted from Tensorflow to Pytorch by work [StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

### Preparing your Data
 you can download [data](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view) provided by [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch?tab=readme-ov-file) and extract it into 
```
data/[yourdataset]
```

### Preparing the moments used for calculating FID
```
python3 calculate_moments.py --data_path data/shells/img --moments_path data/torch_real_moments/shells_moments.npz
```

### Runing the code
```
python3 train.py \
--path data/shells/img --dataset_name shells --moments_path data/torch_real_moments/shells_moments.npz \
--chain --chain_blocks 12345 --tau 0.5 --lbd 20
```

### some experience that could be used for better performance. 
1. You can adjust --chain_blocks to be like 123456, 2345, 23456, 12345, large data requires less blocks
2. 
