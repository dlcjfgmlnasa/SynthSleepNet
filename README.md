# A Foundational Model for Sleep Analysis via Multimodal Self-Supervised Learning Framework

**Full code coming soon^^**

[[`BibTeX`](#license-and-citation-)] 

## Abstract 🔥
Sleep is an essential factor for maintaining human health and quality of life. The analysis of physiological signals during sleep plays a critical role in assessing sleep quality and diagnosing sleep disorders. However, manual diagnosis by clinicians is time-consuming and subjective. While advancements in deep learning have improved automation, they still rely heavily on large-scale labeled data. In this study, we propose SynthSleepNet, a multimodal hybrid self-supervised learning framework for analyzing sleep data (i.e. PSG data). SynthSleepNet effectively combines masked prediction and contrastive learning to integrate complementary features across multiple modalities (i.e., EEG, EOG, EMG, ECG), enabling the model to learn highly expressive representations. Additionally, a temporal context module based on Mamba was developed to efficiently capture contextual information across signals. As a result, SynthSleepNet outperformed state-of-the-art methods in three downstream tasks (i.e., sleep stage classification, apnea detection, hypopnea detection), achieving accuracy of 89.89%, 99.75%, and 89.60%, respectively. The model also demonstrated robust performance in a semi-supervised learning environment with limited labels, achieving accuracy of 87.98%, 99.37%, and 77.52%. The proposed model highlights its potential as a foundation model for comprehensive analysis of sleep data and is expected to set a new standard for sleep disorder monitoring and diagnostic systems.

![synthsleepnet structure](https://github.com/dlcjfgmlnasa/SynthSleepNet/blob/main/figures/architecture.png)

## Usage
### 1. Downloading Dataset
The Sleep Heart Health Study (SHHS) EDF (European Data Format) files are available for download from the [National Sleep Research Resource (NSRR)](https://sleepdata.org/datasets/shhs). NSRR provides access to a variety of sleep study datasets, including SHHS.

### 2. Preprocessing
To convert EDF files into Parquet format using the `dataset/data_parser.py` python script

### 3. Training

#### [Step 1] Pretrained NeuroNet
Each physiological signal (i.e., `EEG` 🧠, `EOG` 👀, `ECG` 💓, `EMG` 💪) was pretrained using *NeuroNet*. *NeuroNet* is a self-supervised learning framework designed for training on single-modality physiological signals. To train each modality, use `pretrained/unimodal/{modality_name}/train.py`.

#### [Step 2] Pretrained SynthSleepNet
*SynthSleepNet* is a multimodal hybrid self-supervised learning framework designed to effectively synthesize information from different physiological signal modalities. To train *SynthSleepNet*, use `pretrained/multimodal/train.py`.

#### [Step 3] Downstream Task
The pretrained *SynthSleepNet* can be used to perform downstream tasks.

1. Linear Probing

    To perform linear probing, use `downstream/linear_probing/train.py`
    
2. Fine-Tuning

    To perform fine-tuning, use `downstream/fine_tuning/train.py`
    

## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code:

- *SynthSleepNet*
```

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
