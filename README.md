# BusterNet: Detecting Copy-Move Image Forgery with Source/Target Localization

### Introduction
We introduce a novel deep neural architecture for image copy-move forgery detection (CMFD), code-named *BusterNet*. Unlike previous efforts, BusterNet is a pure, end-to-end trainable, deep neural network solution. It features a two-branch architecture followed by a fusion module. The two branches localize potential manipulation regions via visual artifacts and copy-move regions via visual similarities, respectively. To the best of our knowledge, this is the first CMFD algorithm with discernibility to localize source/target regions. 

In this repository, we release many paper related things, including

- a pretrained BusterNet model
- custom layers implemented in keras-tensorflow 
- CASIA-CMFD, CoMoFoD-CMFD, and USCISI-CMFD dataset
- python notebook to reproduce paper results 

### Repo Organization
The entire repo is organized as follows:

- Data - host all datasets
  - *CASIA-CMFD
  - *CoMoFoD-CMFD
  - *USCISI-CMFD-Full
  - USCISI-CMFD-Small
- Model - host all model files
- ReadMe.md - this file

Due to the size limit, we can't host all dataset in repo. For those large ones, we host them externally. *indicated dataset requires to be downloaded seperately. Please refer to the document of each dataset for more detailed downloading instructions.

### Python/Keras/Tensorflow
The original model was trained with

- keras.version = 2.0.7
- tensorflow.version = 1.1.0

we also test the repository with 

- keras.version = 2.2.2
- tensorflow.version = 1.8.0

Though small differences may be found, results are in general consistent. 

### Citation
If you use the provided code or data in any publication, please kindly cite the following paper.

    @inproceedings{wu2018eccv,
      title={BusterNet: Detecting Image Copy-Move Forgery With Source/Target Localization},
      author={Wu, Yue, and AbdAlmageed, Wael and Natarajan, Prem},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2018},
      organization={Springer},
    }

### Contact
- Name: Yue Wu
- Email: yue_wu\[at\]isi.edu
