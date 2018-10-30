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

- **Data** - host all datasets
  - *CASIA-CMFD
  - *CoMoFoD-CMFD
  - *USCISI-CMFD
- **Model** - host all model files
- **ReadMe.md** - this file

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


### License
The Software is made available for academic or non-commercial purposes only. The license is for a copy of the program for an unlimited term. Individuals requesting a license for commercial use must pay for a commercial license. 

      USC Stevens Institute for Innovation 
      University of Southern California 
      1150 S. Olive Street, Suite 2300 
      Los Angeles, CA 90115, USA 
      ATTN: Accounting 

DISCLAIMER. USC MAKES NO EXPRESS OR IMPLIED WARRANTIES, EITHER IN FACT OR BY OPERATION OF LAW, BY STATUTE OR OTHERWISE, AND USC SPECIFICALLY AND EXPRESSLY DISCLAIMS ANY EXPRESS OR IMPLIED WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, VALIDITY OF THE SOFTWARE OR ANY OTHER INTELLECTUAL PROPERTY RIGHTS OR NON-INFRINGEMENT OF THE INTELLECTUAL PROPERTY OR OTHER RIGHTS OF ANY THIRD PARTY. SOFTWARE IS MADE AVAILABLE AS-IS. LIMITATION OF LIABILITY. TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT WILL USC BE LIABLE TO ANY USER OF THIS CODE FOR ANY INCIDENTAL, CONSEQUENTIAL, EXEMPLARY OR PUNITIVE DAMAGES OF ANY KIND, LOST GOODWILL, LOST PROFITS, LOST BUSINESS AND/OR ANY INDIRECT ECONOMIC DAMAGES WHATSOEVER, REGARDLESS OF WHETHER SUCH DAMAGES ARISE FROM CLAIMS BASED UPON CONTRACT, NEGLIGENCE, TORT (INCLUDING STRICT LIABILITY OR OTHER LEGAL THEORY), A BREACH OF ANY WARRANTY OR TERM OF THIS AGREEMENT, AND REGARDLESS OF WHETHER USC WAS ADVISED OR HAD REASON TO KNOW OF THE POSSIBILITY OF INCURRING SUCH DAMAGES IN ADVANCE. 

For commercial license pricing and annual commercial update and support pricing, please contact: 

      Rakesh Pandit USC Stevens Institute for Innovation 
      University of Southern California 
      1150 S. Olive Street, Suite 2300
      Los Angeles, CA 90115, USA 

      Tel: +1 213-821-3552
      Fax: +1 213-821-5001 
      Email: rakeshvp@usc.edu and ccto: accounting@stevens.usc.edu
