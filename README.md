# Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-RGB Segmentation 

**NeurIPS 2025**

**Authors:** Nan Bao, Yifan Zhao, Lin Zhu, Jia Li

![Main](https://github.com/iCVTEAM/ESC/blob/master/figs/Main.png)

## Installation

```bash
conda create -n ESC python=3.11.7
conda activate ESC
conda install mkl==2023.1.0 numpy==1.26.3
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python==4.9.0.80 tqdm==4.66.1 pythae==0.1.2 timm==0.9.12 fvcore==0.1.5.post20221221 mmcv==2.1.0 seaborn==0.13.2
```

## Pretrained Models Preparation

1. Download the SegFormer pretrained models and our pretrained edge dictionary models based on VQ-VAE from [here](https://drive.google.com/drive/folders/1lRVSaqPk76T7qAKYGnTi_BF9a7SAKhcb).

2. Move all pretrained models to the `./pretrained` directory.

## Data Preparation

### DERS-XS, DERS-XR, and DSEC-Xtrm

1. Download DERS-XS, DERS-XR, and DSEC-Xtrm from [here](https://drive.google.com/drive/folders/1yyUaPBxUTe1aoi9yVRoFz22TVdY2M5P2).

2. Unzip the files and create symbolic links to the `./data` directory.

    ```bash
    ln -s /path/to/DERS_XS ./data/DERS_XS
    ln -s /path/to/DERS_XR ./data/DERS_XR
    ln -s /path/to/DSEC ./data/DSEC
    ```

### DSEC-Semantic

3. 
    Download the files `event_left.zip` and `image_timestamps.txt` for the following sequences from [the official DSEC website](https://dsec.ifi.uzh.ch/dsec-datasets/download/):
     
    ```
    zurich_city_00_a, zurich_city_01_a, zurich_city_02_a, zurich_city_04_a, zurich_city_05_a, zurich_city_06_a, zurich_city_07_a, zurich_city_08_a, zurich_city_13_a, zurich_city_14_c, zurich_city_15_a
    ```

    Unzip the downloaded files and organize them into the following directory structure:

    ```bash
    .
    ├── DSEC_test
    │   ├── zurich_city_13_a
    │   │   ├── events
    │   │   │   └── left
    │   │   │       ├── events.h5
    │   │   │       └── rectify_map.h5
    │   │   └── images
    │   │   │   └── timestamps.txt
    │   ├── zurich_city_14_c
    │   │   └── ...
    │   └── zurich_city_15_a
    │       └── ...
    └── DSEC_train
        ├── zurich_city_00_a
        │   └── ...
        ├── zurich_city_01_a
        │   └── ...
        └── ...
    ```

4. 
    Update the paths in `./scripts/prepare_dsec_semantic/prepare_dsec_semantic.sh`:

    ```bash
    input_path_train=/path/to/DSEC/DSEC_train
    input_path_test=/path/to/DSEC/DSEC_test
    ```

    Then execute the script:

    ```bash
    bash ./scripts/prepare_dsec_semantic/prepare_dsec_semantic.sh
    ```

    This script processes the raw event data from DSEC-Semantic and organizes the processed data into `./data/DSEC`.

## Evaluating with our Released Models 

1. Download our released models from [here](https://drive.google.com/drive/folders/1aJQ8ZqyjtpXDiaRNxv_1_GYf_0LHeoyZ).

2. Move the downloaded models to the directory: `./ckpt/{task_name}/{model_name}`. Here, {task_name} corresponds to the task name specified in the configuration .yaml files.

3. Run the following commands to evaluate the models:

```bash
python main.py --eval --config-file ./configs/ESC-DERS_XS.yaml --model-name ESC-DERS_XS.pth.tar
python main.py --eval --config-file ./configs/ESC-DERS_XR.yaml --model-name ESC-DERS_XR.pth.tar
python main.py --eval --config-file ./configs/ESC-DSEC_Xtrm.yaml --model-name ESC-DSEC_Xtrm.pth.tar
python main.py --eval --config-file ./configs/ESC-DSEC_Semantic.yaml --model-name ESC-DSEC_Semantic.pth.tar
```

The evaluation results are summarized in the table below:

|Model-Dataset|gACC|mACC|mIoU|
|-|-|-|-|
|ESC-DERS_XS|0.932706|0.752572|0.670984|
|ESC-DERS_XR|0.979211|0.707526|0.652239|
|ESC-DSEC_Xtrm|0.881835|0.594495|0.508670|
|ESC-DERS_Semantic|0.948510|0.786135|0.710352|

## Training and Evaluating Examples

### Training (DDP)

```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --config-file ./configs/ESC-DERS_XS.yaml
```

### Evaluating

```bash
python main.py --eval --config-file ./configs/ESC-DERS_XS.yaml --model-name model-best.pth.tar
```

## Acknowledgement

This code is developed on [SegFormer](https://github.com/NVlabs/SegFormer) and [pythae](https://github.com/clementchadebec/benchmark_VAE). Thanks for these great projects!

## Citation

If you find our work useful for your research, please cite the following paper.

```bib
@inproceedings{
    bao2025recoding,
    title={Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-{RGB} Segmentation},
    author={Bao, Nan and Zhao, Yifan and Zhu, Lin and Li, Jia},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=uG9F00zKJF}
}
```