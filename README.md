# MoPC-HR

 Specific emitter identification (SEI) is a non-password authentication method that adds an extra layer of security to wireless devices. However,
existing SEI methods are unable to continuously learn new classes from a limited number of training examples due to data scarcity, 
which is more challenging than the catastrophic forgetting and overfitting problems associated with the widely studied class-incremental learning (CIL). 
In this paper, we propose a novel few-shot class-incremental specific emitter identification (FSCIL-SEI) framework to address the challenge of catastrophic forgetting and overfitting in CIL.
Specifically, to ensure inter-class discriminability during the incremental process, we first employ prototype learning training methods in the base task and introduce a self-supervised contrastive learning (SSCL) that increases inter-class distances and reduces intra-class distances in the feature space.
Secondly,  we propose a separation of class weights (SCW) to isolate old and new class weights 
in the classification layer, which effectively mitigates the issue of catastrophic forgetting.
 Finally, to alleviate the problem of overfitting due to insufficient samples during incremental training, we introduce a three-stage course learning (CL) approach that advances from simple to complex tasks, which not only mitigates overfitting but also improves the generalization ability of the model. Experimental results demonstrate that our method outperforms other FSCIL methods in terms of both performance degradation (PD) and incremental accuracy when evaluated on automatic identification system (AIS) and automatic dependent surveillance-broadcast (ADS-B) datasets.
## 📁 文件结构说明

| 文件名 | 作者 | 说明 |
|--------|------|------|
| `MoPC_HR_trainer.py` | 自主编写 | 主训练脚本，负责模型训练流程调度 |
| `ads_resnet.py` | 自主编写 | 实现用于 RF 分类的 ResNet 网络结构 |
| `AIS100_1024.py` | 自主编写 / 基于公开数据处理 | 处理 AIS 或 ADS-B 数据集用于训练的预处理脚本 |
| `ConvBlockAttention.py` | 自主编写 | 定义了注意力机制卷积模块，供主网络调用 |


## 📊 数据来源

本项目中使用的ADS=-B数据集来自以下公开论文：

- **论文题目**：Class-Incremental Learning for Wireless Device Identification in IoT  
- **作者**：Y. Liu, J. Wang, J. Li, S. Niu and H. Song  
- **发表期刊**：IEEE Internet of Things Journal  
- **卷期页码**：Vol. 8, No. 23, pp. 17227–17235  
- **发表时间**：2021年12月1日  
- **DOI**：[10.1109/JIOT.2021.3078407](https://doi.org/10.1109/JIOT.2021.3078407)  
- **IEEE引用格式**：  
  > Y. Liu, J. Wang, J. Li, S. Niu and H. Song, "Class-Incremental Learning for Wireless Device Identification in IoT,"  
  > *IEEE Internet of Things Journal*, vol. 8, no. 23, pp. 17227-17235, 1 Dec.1, 2021,  
  > doi: [10.1109/JIOT.2021.3078407](https://doi.org/10.1109/JIOT.2021.3078407)

> ⚠️ 本项目仅供学术用途，数据归原作者所有，请根据原论文引用使用。

 

## 📌 参考项目

本项目参考了以下 GitHub 仓库的部分实现思路或代码结构：

- [G-U-N/PyCIL](https://github.com/G-U-N/PyCIL.git)：一个基于 PyTorch 的增量学习（Class-Incremental Learning）框架，为本项目提供了部分模型结构和训练流程的设计思路。


特别感谢原作者们的开源贡献 🙏

 
