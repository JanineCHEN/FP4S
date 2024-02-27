# FP4S-Floor-plan-image-segmentation-via-scribble-based-semi-weakly-supervised-learning
FP4S: Floor plan image segmentation via scribble-based semi-weakly-supervised learning

Official code and instructions for [**"Floor Plan Image Segmentation Via Scribble-Based Semi-Weakly Supervised Learning: A Style and Category-Agnostic Approach"**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4727643)

The paper has been submitted to **Automation in Construction** and is currently under peer review.

The authors will make the annotated floor plan image datasets, the model code and the model weights of the best performance available for future research soon.

# Introduction
This project introduces a scribble-based semi-weakly-supervised framework for floor plan image segmentation, merging weakly annotated and unlabeled images to boost model robustness and generalizability. This framework benefits from a simplified annotation process while retaining detailed information. Accordingly, we provide a new benchmark dataset for floor plan image parsing covering a wide range of architectural styles and categories.

# Dependencies
- Linux
- Pytorch
- Other required packages are summerized in `requirements.txt`
- CUDA-supported GPU with at least GB memory size is required for training

# Quick start
## Download the repository
```
https://github.com/JanineCHEN/FP4S.git
cd FP4S
```
## Setup virtual environment
Using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/index.html) as an example:
```
mkvirtualenv FP4S
workon FP4S
pip install torch torchvision
pip install -r requirements.txt
```
## Dataset
For downloading the dataset, please refer to <a href="https://github.com/JanineCHEN/FP4S/tree/main/dataset">dataset</a>.


## Training
For training the FP4S model, please run:
```
python main.py
```

## Download the checkpoint
For downloading the checkpoints, please refer to <a href="https://github.com/JanineCHEN/FP4S/tree/main/ckpt">ckpt</a>.

### Acknowledgement
The computational work for this research was performed on resources of the National Supercomputing Centre, Singapore (https://www.nscc.sg). The data sources used in this study are also gratefully acknowledged. This research was supported by the President's Graduate Fellowship of the National University of Singapore.
