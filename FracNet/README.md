# FracNet

## Abstract
**Background**: Diagnosis of rib fractures plays an important role in identifying trauma severity. However, quickly and precisely identifying the rib fractures in a large number of CT images with increasing number of patients is a tough task, which is also subject to the qualification of radiologist. We aim at a clinically applicable automatic system for rib fracture detection and segmentation from CT scans.

**Methods**: A total of 7,473 annotated traumatic rib fractures from 900 patients in a single center were enrolled into our dataset, named RibFrac Dataset, which were annotated with a human-in-the-loop labeling procedure. We developed a deep learning model, named FracNet, to detect and segment rib fractures. 720, 60 and 120 patients were randomly split as training cohort, tuning cohort and test cohort, respectively. FreeResponse ROC (FROC) analysis was used to evaluate the sensitivity and false positives of the detection performance, and Intersection-over-Union (IoU) and Dice Coefficient (Dice) were used to evaluate the segmentation performance of predicted rib fractures. Observer studies, including independent human-only study and human-collaboration study, were used to benchmark the FracNet with human performance and evaluate its clinical applicability. A annotated subset of RibFrac Dataset, including 420 for training, 60 for tuning and 120 for test, as well as our code for model training and evaluation, was open to research community to facilitate both clinical and engineering research.

**Findings**: Our method achieved a detection sensitivity of 92.9% with 5.27 false positives per scan and a segmentation Dice of 71.5% on the test cohort. Human experts achieved much lower false positives per scan, while underperforming the deep neural networks in terms of detection sensitivities with longer time in diagnosis. With human-computer collobration, human experts achieved higher detection sensitivities than human-only or computer-only diagnosis.

**Interpretation**: The proposed FracNet provided increasing detection sensitivity of rib fractures with significantly decreased clinical time consumed, which established a clinically applicable method to assist the radiologist in clinical practice.

For more details, please refer to our paper: 

**Deep-learning-assisted detection and segmentation of rib fractures from CT scans: Development and validation of FracNet**

*Liang Jin\*, [Jiancheng Yang](http://jiancheng-yang.com/)\*, [Kaiming Kuang](http://kaimingkuang.github.io/), [Bingbing Ni](https://scholar.google.com/citations?user=eUbmKwYAAAAJ), Yiyi Gao, Yingli Sun, Pan Gao, Weiling Ma, Mingyu Tan, Hui Kang, Jiajun Chen, Ming Li*

EBioMedicine ([DOI](https://doi.org/10.1016/j.ebiom.2020.103106))

## Code Structure
* FracNet/
    * [`dataset/`](./dataset): PyTorch dataset and transforms.
    * [`models/`](./models): PyTorch 3D UNet model and losses.
    * [`utils/`](./utils): Utility functions.
    * [`main.py`](main.py): Training script.

## Requirements
```
SimpleITK==1.2.4
fastai==1.0.59
fastprogress==0.1.21
matplotlib==3.1.3
nibabel==3.0.0
numpy>=1.18.5
pandas>=0.25.3
scikit-image==0.16.2
torch==1.4.0
tqdm==4.38.0
```

## Usage

### Install Required Packages
First install required packages in [`requirements.txt`](requirements.txt) using pip:
```bash
pip install -r requirements.txt
```
or Anaconda:
```bash
conda install --yes --file requirements.txt
```
To evaluate model predictions, [the official RibFrac-Challenge repository](https://github.com/M3DV/RibFrac-Challenge) is needed. First clone the repository:
```bash
git clone git@github.com:M3DV/RibFrac-Challenge.git <repo_dir>
```
Then change the working directory and install the package:
```bash
cd <repo_dir>
python setup.py install
```

### Download the Dataset
We collect a large-scale rib fracture CT dataset, named RibFrac Dataset as a benchmark for developping algorithms on rib fracture detection, segmentation and classification. You may access the public part of RibFrac dataset via [RibFrac Challenge](https://ribfrac.grand-challenge.org/dataset/) website after one-click free registeration, which was an official MICCAI 2020 challenge. There is slight difference with the public dataset in this paper and that in the RibFrac Challenge, please refer to the [RibFrac Challenge](https://ribfrac.grand-challenge.org/tasks/) website for details.

### Training
To train the FracNet model, run the following in command line:
```bash
python -m main --train_image_dir <training_image_directory> --train_label_dir <training_label_directory> --val_image_dir <validation_image_directory> --val_label_dir <validation_label_directory>
```

### Prediction
To generate prediction, run the following in command line:
```bash
python -m predict --image_dir <image_directory> --pred_dir <predition_directory> --model_path <model_weight_path>
```

In the [predict.py](predict.py), we adopt a post-processing procedure of [removing low-probability regions](predict.py#L18), [spine regions](predict.py#L24), and [small objects](predict.py#L48). This procedure leads to fewer false negatives. You may also skip the post-processing by setting `--postprocess False` in the command line argument and check the raw output.

***Note***: This project aims at a prototype for RibFrac Challenge; However, as the challenge data provider, we would like to avoid unintended data leakage. Therefore, we did not provide full details for the models, including those in both training and inference stages. Nevertheless, it is guaranteed that the performance in the EBioMedicine paper could be reproduced with this one-stage FracNet using 3D UNet as backbone. See more discussion in this [issue](https://github.com/M3DV/FracNet/issues/7).


### Evaluation
To evaluate your prediction, run the following in command line:
```bash
python -m ribfrac.evaluation --gt_dir <gt_directory> -pred_dir <prediction_directory> --clf False
```
