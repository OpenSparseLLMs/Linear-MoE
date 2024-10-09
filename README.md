
<div align="center">

# Linear-MoE

[![hf_model](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/xxx) | [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/xxx)
</div>

This repository offers a **production-ready framework** for modeling and training Linear-MoE models, non-invasively built on the latest [Megatron-Core](https://github.com/NVIDIA/Megatron-LM). It supports state-of-the-art open-source Mixture of Experts (MoE) models, seamlessly integrated with advanced linear sequence modeling techniques such as Linear Attention, State Space Models, and Linear RNNs. **Contributions through pull requests are highly encouraged!**

<!-- <div align="center">
  <img width="400" alt="image" src="https://github.com/xxx">
</div> -->

# Models

|   Linear Sequence Modeling  |  Instance  |  Qwen2 MoE (@Alibaba)  |    Deepseek v2 MoE (@Deepseek)       |    Mixtral MoE (@Mistral AI)   |
| :-----: | :----------------------------: | :----------------------------: | :---------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| Linear Attention |       [Basic Linear Attention](https://arxiv.org/abs/2006.16236) <br> (@Idiap@EPFL)       | ✅ |          ✅          |     ✅      |
|  |       [Retention](https://arxiv.org/abs/2307.08621) <br> (@MSRA@THU)       | ✅ |          ✅          |     ✅      |
|  |         [GLA](https://arxiv.org/abs/2312.06635)  <br> (@MIT@IBM)         | ✅ |     ✅      |    ✅       |
|  |           [Delta Net](https://arxiv.org/abs/2102.11174) <br> (@MIT)            | ✅ |    ✅    |     ✅      |
|  | [Based](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based) <br> (@Stanford@HazyResearch) | ✅ |      ✅      |      ✅     | 
|  |            [Rebased](https://arxiv.org/abs/2402.10644) <br> (@Tinkoff)            | ✅ |  ✅  |      ✅     |
| State Space Modeling (SSM) |             [Mamba2](https://arxiv.org/abs/2405.21060) <br> (@Princeton@CMU)              | ✅ | ✅  |      |
| Linear RNN |             [RWKV6](https://arxiv.org/abs/2404.05892) <br> (@RWKV)              |  ✅  |   ✅   |    ✅    |
|  |             [HGRN2](https://arxiv.org/abs/2404.07904) <br> (@TapTap@Shanghai AILab)             | ✅ |   ✅   |   ✅   |  



# Installation

Your environment should satify the following requirements:

- [PyTorch](https://pytorch.org/) >= 2.0
- [Triton](https://github.com/openai/triton) >=2.2

## Containers
We recommend using the latest release of [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) with DGX nodes, which already have relatively new versions of CUDA, cuDNN, NCCL, PyTorch, Triton, Apex, TransformerEngine, etc., installed.

On the top of NGC's PyTorch container, you can setup Linear-MoE with:
```bash
# Linear-MoE 
git clone --recurse-submodules https://github.com/weigao266/Linear-MoE-public.git

# requirements
pip install -r requirements.txt
```

If you can't use this for some reason, try installing in a Virtualenv.

## Virtualenv

```bash
# create a conda env, install PyTorch
conda create -n linear-moe python=3.11
conda activate linear-moe
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# some nessesary Python packages
pip install six regex packaging

# Transformer Engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Apex
git clone https://github.com/NVIDIA/apex.git
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# (if needed) FlashAttention
MAX_JOBS=8 pip install flash-attn --no-build-isolation

# (if needed) dropout_layer_norm in FlashAttention
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/csrc/layer_norm & pip install .

# Linear-MoE 
git clone --recurse-submodules https://github.com/weigao266/Linear-MoE-public.git

# requirements
pip install -r requirements.txt
```

# Usage

## Pretraining

## Finetuning

## Generation

## Evaluation


# Citation
<!-- If you find this repo useful, please consider citing our works:
```bib

``` -->