# CoMoFoD-CMFD Dataset

### Introduction
This copy-move forgery detection(CMFD) dataset is made upon the original [CoMoFoD dataset](http://www.vcl.fer.hr/comofod/comofod.html) by 

- select copy-move samples only
- obtain the target copy by thresholding the difference between the manipulated image and its original 
- obtain the source copy by matching the target copy on the manipulated image using SIFT/ORB/SURF features
- manually verify all obtained masks

### Folder Content
This CoMoFoD-CMFD dataset folder contains the following things:

* **BusterNetFig6.ipynb** - a python notebook to reproduce Fig6 in the paper
* **BusterNetOnCoMoFoD.ipynb** - a python notebook to 
  - reproduce the results of using the number of correct detections (Table 3)
  - reproduce the BusterNet performance F1 scores
  - show qualitative results on the CoMoFoD-CMFD dataset
* [***CoMoFoD-CMFD.hd5**](https://drive.google.com/file/d/1mefnKzSG6NodumeYWX08XYUHoud7RwZx/view?usp=sharing) - the HDF dataset of CoMoFoD-CMFD samples
  - both images and masks are included
  - data have already been preprocessed
  - see sample usage in **BusterNetOnCoMoFoD.ipynb**
  - *NOT included, right click to download the dataset from Google drive
* **GT_Mask.tar.gz** - the gzip tar file of CMFD 3-class masks
* **ReadMe.md** - this file
