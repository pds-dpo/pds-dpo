# PDS-DPO: Multimodal Preference Data Synthetic Alignment with Reward Model
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.17417-b31b1b.svg)](https://arxiv.org/abs/2412.17417)&nbsp;
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

## News
* **2024-12:** 📃 Our paper is accesible at [arXiv](https://arxiv.org/abs/2412.17417) now!
* **2024-12:** 🚀 We open-source the code, weights ([7B](https://huggingface.co/pdsdpo/PDS-DPO-7B), [7B-LoRA](https://huggingface.co/pdsdpo/PDS-DPO-7B-LoRA)) and [dataset](https://huggingface.co/datasets/pdsdpo/pdsdpo-v1_0-data) of PDS-DPO!


## Installation
```
git clone https://github.com/pds-dpo/pds-dpo.git
cd pds-dpo
conda create -n pdsdpo python=3.10 -y
conda activate pdsdpo
pip install --upgrade pip
pip install -e .
```

You may skip step 1 and step 2 and proceed to step 3 directly as we have provided the resulting dataset in our [HuggingFace](https://huggingface.co/datasets/pdsdpo/pdsdpo-v1_0-data). 
## Step 1: Image Generation and Ranking 
We have provide the sample text-to-image prompts in ```prompt/sample.txt```. You can run the generation and ranking script directy as follows.
```
cd image_generation_ranking
python run.py
```
All images are stored in the ```images``` folder. For each prompt, the script produces four images, which are saved in the ```sample``` folder. The image with the highest ranking score is selected and saved separately in the ```sample-ranked``` folder. 
## Step 2: Response Generation and Ranking
We have provide the sample images and instruction prompts in ```instruction-prompts/sample.txt``` and ```images-ranked```, respectively. By default, we utilized four different open-source MLLMs, including llava-v1.6-mistral-7b-, llava-v1.6-vicuna-13b, and llava-v1.6-vicuna-7b. You may modify accordingly with your preference MLLMs.

You can generate the response by simply use this command.
```
cd response_generation_ranking
python run.py
```
The output is the chosen and rejected conversation save as ```output.json``` with the following format:
```
[
    {
        "id": "transport-919",
        "image": "images/transport-919.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image> What challenges does a ferryboat face as it crosses a turbulent sea, with passengers bracing against the spray and wind?"
            },
            {
                "from": "gpt",
                "value": "chosen response"
            }
        ],
        "rejected_conversations": [
            {
                "from": "human",
                "value": "<image> What challenges does a ferryboat face as it crosses a turbulent sea, with passengers bracing against the spray and wind?"
            },
            {
                "from": "gpt",
                "value": "rejected response"
            }
        ]
    }
]
```

## Step 3: MLLM Training with DPO
1. Modify the ```dpo_trainer.py``` in the trl library

To enable image token processing for DPO training, navigate to the trl library directory in your virtual environment: ```cd ./envs/pdsdpo/lib/python3.10/site-packages/trl/trainer/```. Replace ```dpo_trainer.py``` with the provided file from the ```tool``` folder.

2. Prepare the dataset

Download and extract the entire dataset from [HuggingFace](https://huggingface.co/datasets/pdsdpo/pdsdpo-v1_0-data), then save it in the ```data``` folder.

3. Run DPO training

Double-check the parameters on ```scripts/run_dpo.sh``` and made adjustment if necessary, then simply train the model with this command:
```
cd scripts
bash run_dpo.sh
```
We trained the model using 2 x 80GB A100 GPUs.

For comprehensive tutorials on evaluating other benchmarks, please refer to the [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) repository documentation.
## License
This project incorporates specific datasets and checkpoints, each governed by their respective original licenses. Users are required to adhere to the terms and conditions outlined in these licenses. The project’s content is independently licensed under the [Apache license 2.0](https://github.com/pds-dpo/pds-dpo/blob/main/LICENSE).

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
