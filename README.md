# 模型名称 Efficient Non-Sampling Factorization Machines for Optimal Context-Aware Recommendation
## 1. 简介
ENSFM 是一个只有一层预测层的浅 FM 模型，跟 DeepFM, CFM 相比在复杂度和参数量上都更少，却在模型效果上表现显著的优势。结果验证了论文的观点：负采样策略并不足以使模型收敛到最优。与之相比，非采样学习对于优化 Top-N 推荐任务是非常有效的
## 2. 复现精度
- 论文精度 (HR@5：0.0601，HR@10：0.1024，HR@20：0.1690)
- 复现 (HR@5：0.0599，HR@10：0.1002，HR@20：0.1689) 
- 官方代码 (HR@5：0.0594，HR@10：0.1012，HR@20：0.1687)  
复现精度比论文差一点，但是稍优于论文官方代码结果
## 3. 数据集
Movielens
## 4. 环境依赖
paddlepaddle-gpu=2.2.0
## 6. 训练评估
### 复现代码
```
unzip data/ml-1m/train.csv.zip -d data/ml-1m/ 
python ENSFM.py
```
### 官方代码
```
unzip ENSFM-tf/data/ml-1m/train.csv.zip -d ENSFM-tf/data/ml-1m/
cd ENSFM-tf/code/
python ENSFM.py
```
### 日志
- [复现日志](./data/ml-1m/ENSFM.txt)
- [官方代码日志](./ENSFM-tf/data/ml-1m/ENSFM.txt)
