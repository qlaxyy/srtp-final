# SEAL: Steerable Reasoning Calibration of Large Language Models for Free

[Paper Link](https://arxiv.org/abs/2504.07986)

## Abstract

Large Language Models (LLMs), such as OpenAI's o1-series have demonstrated compelling capabilities for complex reasoning tasks via the extended chain-of-thought (CoT) reasoning mechanism. However, recent studies reveal substantial redundancy in the CoT reasoning traces, which not only increases inference latency but also negatively impacts model performance by diverting attention to unnecessary reasoning paths. 

To address this issue, we investigate the internal reasoning structures of LLMs and categorize them into three primary thought types: execution, reflection, and transition thoughts. Moreover, our analysis reveals that excessive reflection and transition thoughts are strongly correlated with failure cases and these thought categories exhibit clear separation in the latent space. 

Based on these, we introduce SEAL (Steerable reasoning calibration), a training-free approach that seamlessly calibrates the CoT process, improving accuracy while demonstrating significant efficiency gains. SEAL consists of an offline stage for extracting the reasoning steering vector in the latent space, followed by an on-the-fly calibration of the reasoning trace through representation intervention using the steering vector. Notably, the steering vector exhibits strong transferability across various tasks. 

Extensive experiments across multiple models (DeepSeek-R1-Distill and QwQ-32B-Preview) and benchmarks (Math500, GSM8K, LiveCodeBench) validate the effectiveness of SEAL, up to a 11% improvement in accuracy while reducing reasoning tokens by 11.8% to 50.4%. Our code is publicly available at this https URL.
