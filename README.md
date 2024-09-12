# FP4S-Floor-plan-image-segmentation-via-scribble-based-semi-weakly-supervised-learning
FP4S: Floor plan image segmentation via scribble-based semi-weakly-supervised learning

Official code and instructions for [**"Floor Plan Image Segmentation Via Scribble-Based Semi-Weakly Supervised Learning: A Style and Category-Agnostic Approach"**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4727643)

The authors will make the annotated floor plan image datasets, the model code and the model weights of the best performance available for future research soon.

# Introduction
This project introduces a scribble-based semi-weakly-supervised framework for floor plan image segmentation, merging weakly annotated and unlabeled images to boost model robustness and generalizability. This framework benefits from a simplified annotation process while retaining detailed information. Accordingly, we provide a new benchmark dataset for floor plan image parsing covering a wide range of architectural styles and categories.

# Dependencies
- Linux
- [Pytorch 1.13.1+cu117](https://pytorch.org/get-started/previous-versions/#v1131)
- Other required packages are summerized in `requirements.txt`
- CUDA-supported GPU with at least 24 GB memory size is required for training, and at least 6 GB memory size is required for inference.

# Quick start
## Download the repository
```
git clone https://github.com/JanineCHEN/FP4S.git
cd FP4S
```
## Setup virtual environment
Using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/index.html) as an example:
```
mkvirtualenv FP4S
workon FP4S
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
## Dataset
For downloading the dataset, please refer to <a href="https://github.com/JanineCHEN/FP4S/tree/main/dataset">dataset</a>.


## Training
For training the FP4S model, please run:
```
python main.py
```
For customized configuration, please refer to `tools/config.py`:
If you want to use cyclic learning rate schedular for training, please run:
```
python main.py --lr_schedule True
```
If you want to normalize the floor plan images for training, please run:
```
python main.py --ifNorm True
```
If you want to leverage pre-trained model weights of the chosen backbone for training, please run:
```
python main.py --ifpretrain True
```
For using either Focal loss or abCE loss for training, please run:
```
python main.py --useFocalLoss True
```
or
```
python main.py--useabCELoss True
```

If you want to use cutmix augmentation, please run:
```
python main.py --cutmix True
```
If you want to change the backbone, please run:
```
python main.py --backbone <the chosen backbone name>
```
If you want to change the batchsize, please run (notice that if cutmix is set to `True`, batchsize will be set to 2 as default):
```
python main.py --batchsize <your preferred batchsize>
```

## Download the checkpoint
For downloading the checkpoints, please refer to <a href="https://github.com/JanineCHEN/FP4S/tree/main/ckpt">ckpt</a>.

### Acknowledgement
The computational work for this research was performed on resources of the National Supercomputing Centre, Singapore (https://www.nscc.sg). The data sources used in this study are also gratefully acknowledged. This research was supported by the President's Graduate Fellowship of the National University of Singapore.
