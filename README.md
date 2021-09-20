# GTM-Transformer
Official Pytorch Implementation of **Well Googled is Half Done: Multimodal Forecasting of New FashionProduct Sales with Image-based Google Trends** paper

## Installation

We suggest the use of VirtualEnv. Requirements file is available.

```bash

python3 -m venv gtm_venv
source gtm_venv/bin/activate

pip install -r requirements.txt

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/HumaticsLAB/GTM-Transformer.git
cd GTM-Transformer
mkdir ckpt
mkdir dataset
mkdir results

unset INSTALL_DIR
```

## Dataset

**VISUELLE** dataset is publicly available to download [here](https://drive.google.com/file/d/1yEZPTUsOuygokMdmpt3RMhFoIkkf2ohY/view?usp=sharing). Please download and extract it inside the dataset folder.

## Training
To train the model of GTM-Transformer please use the following scripts. Please check the arguments inside the script before launch.

```bash
python train.py --data_folder dataset
```


## Inference
To evaluate the model of GTM-Transformer please use the following script .Please check the arguments inside the script before launch.

```bash
python forecast.py --data_folder dataset --ckpt_path ckpt/model.pth
```

## Citation
```
BibTex
```
