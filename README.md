# NeuroImageRegistration [![Build Status](https://travis-ci.org/Danielhiversen/NeuroImageRegistration.svg?branch=master)](https://travis-ci.org/Danielhiversen/NeuroImageRegistration)

## Install
http://neuro.debian.net/install_pkg.html?p=fsl-complete

```
sudo apt-get install -y libblas-dev liblapack-dev libfreetype6-dev
sudo apt-get install -y cmake ninja-build git
sudo apt-get install gfortran

git clone git://github.com/stnava/ANTs.git
mkdir antsbin
cd antsbin
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ../ANTs/
ninja

sudo apt-get install python-pip
cd
git clone git@github.com:Danielhiversen/NeuroImageRegistration.git
cd NeuroImageRegistration/
virtualenv venv
source venv/bin/activate
sudo pip install --upgrade setuptools
sudo pip install --upgrade distribute
pip install -r requirements.txt
```

ant registration parameters inspired by
http://miykael.github.io/nipype-beginner-s-guide/normalize.html
https://apps.icts.uiowa.edu/confluence/display/BRAINSPUBLIC/ANTS+conversion+to+antsRegistration+for+same+data+set


## Usage
Some computer specific pathes must be sat in util.py

- ConvertDataToDB.py, 
- ConvertDataToDB_menigiomer.py and 
- ConvertDataToDB_MM.py 

are used to convert the data and add it to a database

- do_img_registration.py,	
- do_img_registration_GBM.py, 
- do_img_registration_LGG_POST.py,	
- do_img_registration_LGG_PRE.py	and 
- do_img_registration_MM.py

are used do registration of the volume to a reference volume.


- run_neuro_reg.sh, 
- run_neuro_reg_gbm.sh, 
- run_neuro_reg_gbm2.sh, 
- run_neuro_reg_gbm3.sh and 
- run_neuro_reg_gbm_post.sh

are used to do the image registration on a super computer 

- postProcess.py,	
- postProcess_GBM.py and 
- postProcess_MM.py

are used post processing and generating figures
