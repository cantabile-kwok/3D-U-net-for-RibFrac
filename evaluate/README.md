# RibFrac-Challenge

Evaluation scripts for [MICCAI 2020 RibFrac Challenge: Rib Fracture Detection and Classification](https://ribfrac.grand-challenge.org/).

* The "master" branch is evaluation code for the online evaluation at present, including detection, classification and segmentation metrics. 
* The "legacy" branch is used for the official MICCAI 2020 challenge evaluation before Oct 4, 2020, where fewer evaluation metrics are available. 

# Content

```
RibFrac-Challenge/
    requirements.txt                Required packages for evaluation
    ribfrac/
        evaluation.py               Functions for model evaluation
        nii_dataset.py              The dataset class for .nii reading
```

# Setup

## Install required packages

Run the following in command line to install the required packages. First create a specific Anaconda environment and activate it:
```bash
conda create -n ribfrac python=3.7
conda activate ribfrac
```
And then install required packages using pip:
```bash
pip install -r requirements.txt
```

## Install this repo as a package (optional)

You can also install this repository as a package in your environment if you want to:
```bash
python setup.py install
```

# Usage

## Download the competition data

You can download the competition data by first [Join](https://ribfrac.grand-challenge.org/participants/registration/create/) the challenge then visit the [Dataset](https://ribfrac.grand-challenge.org/dataset/) page.

## Evaluation

We use this script to evaluate your test set submission online. You can evaluate your own prediction locally as well. The evaluation script has very specific requirements on the submission format. Please make sure that these requirements are followed or your submission won't be graded.

To evaluate your prediction locally, you need to prepare the ground-truth and prediction directory. Take validation dataset as an example. After the train/validation data is downloaded, you should unzip it and place all ground-truth label .nii.gz files along with the info .csv file under the same directory as follows:
```
ground_truth_directory/
    ribfrac-val-info.csv
    RibFrac421-label.nii.gz
    RibFrac422-label.nii.gz
    ...
    RibFrac500-label.nii.gz

```

Your prediction should be organized similarly to the ground-truth directory. .nii.gz and .csv should be placed under the same directory as follows:
```
prediction_directory/
    ribfrac-val-pred.csv
    RibFrac421.nii.gz
    RibFrac422.nii.gz
    ...
    RibFrac500.nii.gz
```

Each .nii.gz file should contain a 3D volume with ```n``` fracture regions labelled in integer from ```1``` to ```n```. The order of axes should be ```(x, y, z)``` (something like 512 x 512 x 381).

The prediction info .csv should have four columns: ```public_id``` (patient ID), ```label_id``` (prediction ID marking the specific connected-region in the .nii volume), ```confidence``` (detection confidence) and ```label_code``` (fracture class), e.g.:

|public_id|label_id|confidence|label_code|
|-|-|-|-|
|RibFrac421|0|0.5|0|
|RibFrac421|1|0.5|2|
|RibFrac422|0|0.5|0|
|RibFrac422|1|0.5|3|
|...||||
|RibFrac500|0|0.5|0|
|RibFrac500|1|0.5|3|

For each public_id, there should be at least one row representing the background class. Similar to the ground-truth info .csv, the background record should have ```label_id=0``` and ```label_code=0```. Other than that, each row in the classification prediction .csv represents one predicted fracture area. The public_id should be in the same format as in .nii file names.

You could refer to submission samples for the [validation set](https://drive.google.com/file/d/1o87IKN2MN6ty__V0tc079BvTAPb_uT-K/view) and [test set](https://drive.google.com/file/d/1ehEo69m7hsKG8G_Q5TSLfl8x5y-OepeB/view). Please note that submission samples are randomly generated to pass the evaluation sanity check. 

After setting all of the above, you can evaluate your prediction through the following command line:
```bash
python -m ribfrac.evaluation --gt_dir <ground_truth_directory> --pred_dir <prediction_directory>
```
