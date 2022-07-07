# craft.pytorch
This is a replication of CRAFT(Character-Region Awareness For Text Detection) with training code example.


## Requirement
- Anaconda
- python 3.8
- pytorch 1.9
- pytorch-lightning 1.4.8
## Setup Environment
To run this repository you need to install and activate the environment from yaml file using anaconda with this command:
```
conda env create -f environment.yml
conda activate craft
```

## Train The Network
To train the network you can run the script below :
```
python train.py --max_epoch 10 --dataset_path /data/SynthText --dataset_type synthtext --resume weights/craft_mlt_25k.pth
```