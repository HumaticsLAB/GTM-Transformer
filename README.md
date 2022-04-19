# GTM-Transformer
Official Pytorch Implementation of [**Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends**](https://arxiv.org/abs/2109.09824) paper

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/well-googled-is-half-done-multimodal/new-product-sales-forecasting-on-visuelle)](https://paperswithcode.com/sota/new-product-sales-forecasting-on-visuelle?p=well-googled-is-half-done-multimodal)

## Installation

We suggest the use of VirtualEnv.

```bash

python3 -m venv gtm_venv
source gtm_venv/bin/activate
# gtm_venv\Scripts\activate.bat # If you're running on Windows

pip install numpy pandas matplotlib opencv-python permetrics Pillow scikit-image scikit-learn scipy tqdm transformers fairseq wandb

pip install torch torchvision

# For CUDA11.1 (NVIDIA 3K Serie GPUs)
# Check official pytorch installation guidelines for your system
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

**VISUELLE** dataset is publicly available to download [here](https://forms.gle/cVGQAmxhHf7eRJ937). Please download and extract it inside the dataset folder.

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
@misc{skenderi2021googled,
      title={Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends}, 
      author={Geri Skenderi and Christian Joppi and Matteo Denitto and Marco Cristani},
      year={2021},
      eprint={2109.09824},
}
```
