# CASIA-CMFD Dataset

### Introduction
This copy-move forgery detection(CMFD) dataset is made upon the original [CASIA-Tide-V2 dataset](http://forensics.idealtest.org/casiav2/) by 

- select copy-move samples only
- obtain the target copy by thresholding the difference between the manipulated image and its original 
- obtain the source copy by matching the target copy on the manipulated image using SIFT/ORB/SURF features
- manually verify all obtained masks

In the end, we have 1313 positive CMFD samples (i.e. with copy-move forgery). The corresponding original images are used as negative samples. It is worth to mention that there are 991 unique negative samples, because some positive samples point to the same original image.

### Folder Content
This CASIA-CMFD dataset folder contains the following things:

* **BusterNetOnCASIA.ipynb** - a python notebook to 
  - reproduce the results of using the pixel evaluation protocol-B (Table 2)
  - reproduce discernibility analysis results (Sec 5.5 in paper)
  - show qualitative results on the CASIA-CMFD dataset
* **CASIA-CMFD-Pos.hd5** - the HDF dataset of positive CASIA-CMFD samples
  - both images and masks are included
  - data have already been preprocessed
  - see sample usage in **BusterNetOnCASIA.ipynb**
* **GT_Mask.tar.gz** - the gzip tar file of raw positive masks
* **negative_samples.ids** - the 1313 CASIA ids of negative samples
* **negative_samples.unique.ids** - the 991 unique CASIA ids of negative samples
* **positive_samples.ids** - the 1313 CASIA ids of positive samples
* **ReadMe.md** - this file