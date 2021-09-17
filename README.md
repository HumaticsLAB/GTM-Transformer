# GTM-Transformer
Official Implementation of **Well Googled is Half Done: Multimodal Forecasting of New FashionProduct Sales with Image-based Google Trends** paper

## Installation

We suggest the use of VirtualEnv. Requirements file is available.

```bash

python3 -m venv gtm_venv
source gtm_venv/bin/activate

pip install -r requirements.txt

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/VIPS4/GTM-Transformer.git
cd GTM-Transformer

unset INSTALL_DIR
```

## Dataset

**VISUELLE** dataset is publicly available to download [here](.)

## Training
To train the model of GTM-Transformer please use the following scripts. Please check the arguments inside the script before launch.

```bash
python train.py
```


## Inference
To evaluate the model of GTM-Transformer please use the following script .Please check the arguments inside the script before launch.

```bash
python forecast.py
```

## Citation
```
BibTex
```
