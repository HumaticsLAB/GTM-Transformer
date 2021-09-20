# GTM-Transformer
Official Pytorch Implementation of **Well Googled is Half Done: Multimodal Forecasting of New FashionProduct Sales with Image-based Google Trends** paper

## Installation

We suggest the use of VirtualEnv.

```bash

python3 -m venv gtm_venv
source gtm_venv/bin/activate
# gtm_venv\Scripts\activate.bat # If you're running on Windows

pip install numpy pandas matplotlib opencv-python permetrics Pillow scikit-image scikit-learn scipy tqdm transformers fairseq wandb

pip install torch torchvision

#For CUDA11.1 (NVIDIA 3K Serie GPUs)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install pytorch-lightning

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

**VISUELLE** dataset is publicly available to download [here](https://drive.google.com/file/d/11Bn2efKfO_PbtdqsSqj8U6y6YgBlRcP6/view?usp=sharing). Please download and extract it inside the dataset folder.

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
