# PDS-DPO: Multimodal Preference Data Synthetic Alignment with Reward Model
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](#)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-pdsdpo/PDS--DPO--7B-yellow)](https://huggingface.co/pdsdpo/PDS-DPO-7B)&nbsp;
[![huggingface dataset](https://img.shields.io/badge/%F0%9F%93%81%20Dataset-pdsdpo/pdsdpo--v1_0--data-blue)](https://huggingface.co/datasets/pdsdpo/pdsdpo-v1_0-data)&nbsp;

</div>

<p align="center" style="font-size: larger;">
  <a href="#">Multimodal Preference Data Synthetic Alignment with Reward Model</a>
</p>

### 🔥 Introducing PDS-DPO: a new pipeline in generating preferenced data synthetic with reward model for effective Multimodal LLMs alignment ✨

Starting with an initial text-to-image prompt, the Stable Diffusion model generates synthetic images. These images are then filtered using a reward model to exclude low-quality samples and retain only those with the highest scores. The selected images, along with their corresponding instruction prompts, serve as input for open-source MLLMs to generate responses. These responses are evaluated based on various criteria, and only the highest-scoring ones are selected to identify the most suitable positive and negative pairs for DPO-based training.

<p align="center">
<img src="https://github.com/pds-dpo/pds-dpo/blob/main/assets/pipeline.png" width=93%>
<p>


## Installation

Note: Step 1 and step 2 are optional as we have provided the resulting dataset [![huggingface dataset](https://img.shields.io/badge/%F0%9F%93%81%20Dataset-pdsdpo/pdsdpo--v1_0--data-blue)](https://huggingface.co/datasets/pdsdpo/pdsdpo-v1_0-data)&nbsp;. You may proceed to step 3 directly for reproduction.
## Step 1: Image Generation and Ranking 

## Step 2: Response Generation and Ranking

## Step 3: MLLM Training with DPO

## License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses. The content of this project itself is licensed under the [Apache license 2.0](https://github.com/pds-dpo/pds-dpo/blob/main/LICENSE).

## Citation
```
@Article{PDSDPO,
      title={Multimodal Preference Data Synthetic Alignment with Reward Model}, 
      author={Robert Wijaya and Ngoc-Bao Nguyen and Ngai-Man Cheung},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgement
This research benefits from [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), [ImageReward](https://github.com/THUDM/ImageReward), and [RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling/). Thanks for their great work.
