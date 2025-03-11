# Toward Foundational Model for Sleep Analysis Using a Multimodal Hybrid Self-Supervised Learning Framework

**Full code coming soon^^**

[[`Arvix`](https://arxiv.org/abs/2502.17481)] [[`BibTeX`](#license-and-citation-)] 

## Abstract 🔥
Sleep is essential for maintaining human health and quality of life. Analyzing physiological signals during sleep is critical in assessing sleep quality and diagnosing sleep disorders. However, manual diagnoses by clinicians are time-intensive and subjective. Despite advances in deep learning that have enhanced automation, these approaches remain heavily dependent on large-scale labeled datasets. This study introduces SynthSleepNet, a multimodal hybrid self-supervised learning framework designed for analyzing polysomnography (PSG) data. ***SynthSleepNet effectively integrates masked prediction and contrastive learning to leverage complementary features across multiple modalities, including electroencephalogram (EEG), electrooculography (EOG), electromyography (EMG), and electrocardiogram (ECG). This approach enables the model to learn highly expressive representations of PSG data.*** Furthermore, a temporal context module based on Mamba was developed to efficiently capture contextual information across signals. ***SynthSleepNet achieved superior performance compared to state-of-the-art methods across three downstream tasks***: sleep-stage classification, apnea detection, and hypopnea detection, with accuracies of 89.89%, 99.75%, and 89.60%, respectively. The model demonstrated robust performance in a semi-supervised learning environment with limited labels, achieving accuracies of 87.98%, 99.37%, and 77.52% in the same tasks. These results underscore the potential of the model as a foundational tool for the comprehensive analysis of PSG data. SynthSleepNet demonstrates comprehensively superior performance across multiple downstream tasks compared to other methodologies, making it expected to set a new standard for sleep disorder monitoring and diagnostic systems.

![synthsleepnet structure](https://github.com/dlcjfgmlnasa/SynthSleepNet/blob/main/figures/architecture.png)

## Installation 💿
The code requires `python>=3.9`, as well as `pytorch>=2.0.1` and `torchvision>=0.15`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Clone the repository locally and install with *SynthSleepNet*

```bash
>> git clone https://github.com/dlcjfgmlnasa/SynthSleepNet.git
>> cd SynthSleepNet
>> pip install -r requirements.txt
```


## Usage 🤖

### 1. Downloading Dataset
The Sleep Heart Health Study (SHHS) EDF (European Data Format) files are available for download from the [National Sleep Research Resource (NSRR)](https://sleepdata.org/datasets/shhs). NSRR provides access to a variety of sleep study datasets, including SHHS.

### 2. Preprocessing
To convert EDF files into Parquet format using the `dataset/data_parser.py` python script

### 3. Training

#### [Step 1] Training NeuroNet
Each physiological signal (i.e., `EEG` 🧠, `EOG` 👀, `ECG` 💓, `EMG` 💪) was pretrained using *NeuroNet*. *NeuroNet* is a self-supervised learning framework designed for training on single-modality physiological signals. 

To train *NeuroNet* using each modality, use `pretrained/unimodal/{modality_name}/train.py`.

#### [Step 2] Training SynthSleepNet
*SynthSleepNet* is a multimodal hybrid self-supervised learning framework designed to effectively synthesize information from different physiological signal modalities. 

To train *SynthSleepNet*, use `pretrained/multimodal/train.py`.

#### [Step 3] Downstream Task
The pretrained *SynthSleepNet* can be used to perform downstream tasks.

1. **Linear Probing** : To perform linear probing, use `downstream/linear_probing/train.py`
    
2. **Fine-Tuning** : To perform fine-tuning, use `downstream/fine_tuning/train.py`

## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code

- *SynthSleepNet*
```
@article{lee2025toward,
  title={Toward Foundational Model for Sleep Analysis Using a Multimodal Hybrid Self-Supervised Learning Framework},
  author={Lee, Cheol-Hui and Kim, Hakseung and Yoon, Byung C and Kim, Dong-Joo},
  journal={arXiv preprint arXiv:2502.17481},
  year={2025}
}
```

- *NeuroNet*
```
@article{lee2024neuronet,
  title={NeuroNet: A Novel Hybrid Self-Supervised Learning Framework for Sleep Stage Classification Using Single-Channel EEG},
  author={Lee, Cheol-Hui and Kim, Hakseung and Han, Hyun-jee and Jung, Min-Kyung and Yoon, Byung C and Kim, Dong-Joo},
  journal={arXiv preprint arXiv:2404.17585},
  year={2024}
}
```
