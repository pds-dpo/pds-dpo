# PDS-DPO: Multimodal Preference Data Synthetic Alignment with Reward Model
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](#)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-pdsdpo/PDS--DPO--7B-yellow)](https://huggingface.co/pdsdpo/PDS-DPO-7B)&nbsp;
[![huggingface dataset](https://img.shields.io/badge/%F0%9F%93%81%20Dataset-pdsdpo/pdsdpo--v1_0--data-blue)](https://huggingface.co/datasets/pdsdpo/pdsdpo-v1_0-data)&nbsp;

</div>

<p align="center" style="font-size: larger;">
  <a href="#">Multimodal Preference Data Synthetic Alignment with Reward Model</a>
</p>

### 🔥 Introducing PDS-DPO: a new framework in generating preferenced data synthetic with reward model for effective Multimodal LLMs alignment ✨

Starting with an initial text-to-image prompt, the Stable Diffusion model generates synthetic images. These images are then filtered using a reward model to exclude low-quality samples and retain only those with the highest scores. The selected images, along with their corresponding instruction prompts, serve as input for open-source MLLMs to generate responses. These responses are evaluated based on various criteria, and only the highest-scoring ones are selected to identify the most suitable positive and negative pairs for DPO-based training.

<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/3e12655c-37dc-4528-b923-ec6c4cfef178" width=93%>
<p>
