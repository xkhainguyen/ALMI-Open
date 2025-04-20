# ALMI-X Transformer Foundation Model Training Guide  

## 1. Download ALMI-X Dataset  
The ALMI-X dataset is available on our HuggingFace project:  
```
https://huggingface.co/datasets/TeleEmbodied/ALMI-X
```

## 2. Dataset and Pre-trained Models Preparation  
### Dataset Setup  
1. Place the downloaded ALMI-X dataset in `./dataset/ALMI/`  
2. Extract the files to obtain the following directory structure:  
```
./dataset/ALMI  
│── actions/  
│── texts/  
└── train_ALMI.txt
```
- The `actions` and `texts` directories can be extracted from `data.tar.gz` and `texts.tar.gz` respectively.

### Pre-trained Models  
- Download the CLIP model from [OpenAI's official repository](https://github.com/openai/CLIP) and place it in the `pretrained` directory.  
- Alternatively, you can load the CLIP model directly using:  
```python
clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)
```

## 3. Environment Setup  
```bash
conda env create -f environment.yml
conda activate ALMI_trans
```

## 4. Training Instructions  
The model can be trained on an A100-40G GPU.  

### 4.1 Train cl-20sl Model  
```bash
./train_almi_cl_20sl.sh
```

### 4.2 Train cl-400sl Model  
```bash
./train_almi_cl_400sl.sh
```

### 4.3 Train ol Model  
```bash
./train_almi_vq.sh
./train_almi_ol.sh
```

## 5. Evaluation  

### 5.1 Export Trained Model  

You can use this script to export your trained checkpoint for mujoco:

```bash
python export.py \
    --model_path output/test_cl_20sl/almi_trans_cl_20sl_last.pth \
    --export_path output/export \
    --export_name trans_net_export
```

#### 5.2 Play trained transformer model in mujoco

You can download our trained checkpoints after exporting at: <https://huggingface.co/TeleEmbodied/ALMI-trans>

Create a new conda environment:
```
conda create -n almi-rl python==3.8 # same as the ALMI_EL env
conda activate almi-rl
cd L2L_trans
pip install -r requirements_rl.txt

# see the h1_2_21dof_trans.yaml to change the parameters to satisfy your checkpoint.
python deploy/deploy_mujoco/deploy_mujoco_21dof_trans_clean.py
```
Then you will see the result in mujoco environment.


## 6. Acknowledgements  
We gratefully acknowledge the contribution from:  
- [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)