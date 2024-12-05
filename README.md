# A Foundational Model for Sleep Analysis via Multimodal Self-Supervised Learning Framework

[[`BibTeX`](#license-and-citation-)] 

## Abstract 🔥
Sleep is an essential factor for maintaining human health and quality of life. The analysis of physiological signals during sleep plays a critical role in assessing sleep quality and diagnosing sleep disorders. However, manual diagnosis by clinicians is time-consuming and subjective. While advancements in deep learning have improved automation, they still rely heavily on large-scale labeled data. In this study, we propose SynthSleepNet, a multimodal hybrid self-supervised learning framework for analyzing sleep data (i.e. PSG data). SynthSleepNet effectively combines masked prediction and contrastive learning to integrate complementary features across multiple modalities (i.e., EEG, EOG, EMG, ECG), enabling the model to learn highly expressive representations. Additionally, a temporal context module based on Mamba was developed to efficiently capture contextual information across signals. As a result, SynthSleepNet outperformed state-of-the-art methods in three downstream tasks (i.e., sleep stage classification, apnea detection, hypopnea detection), achieving accuracy of 89.89%, 99.75%, and 89.60%, respectively. The model also demonstrated robust performance in a semi-supervised learning environment with limited labels, achieving accuracy of 87.98%, 99.37%, and 77.52%. The proposed model highlights its potential as a foundation model for comprehensive analysis of sleep data and is expected to set a new standard for sleep disorder monitoring and diagnostic systems.

![synthsleepnet structure](https://github.com/dlcjfgmlnasa/SynthSleepNet/blob/main/figures/architecture.png)

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
