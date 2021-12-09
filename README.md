# 模型名称 Efficient Non-Sampling Factorization Machines for Optimal Context-Aware Recommendation
## 1. 简介
ENSFM 是一个只有一层预测层的浅 FM 模型，跟 DeepFM, CFM 相比在复杂度和参数量上都更少，却在模型效果上表现显著的优势。结果验证了论文的观点：负采样策略并不足以使模型收敛到最优。与之相比，非采样学习对于优化 Top-N 推荐任务是非常有效的
## 2. 复现精度
**复现精度比论文差一点，但是稍优于论文官方代码结果**

| Movielens   |HR@5|   HR@10   |HR@20@5|
| ---- | ----  |  ---- | ----  | 
| 论文  | 0.0601 |0.1024|0.1690|
| 复现  | 0.0599 |0.1002|0.1689|
| 官方代码  | 0.0594 |0.1012|0.1687|

## 3. 环境依赖
paddlepaddle-gpu=2.2.0
## 4. 训练评估
### 复现代码
```
unzip data/ml-1m/train.csv.zip -d data/ml-1m/ 
python ENSFM.py
```
### 官方代码
[链接](https://github.com/chenchongthu/ENSFM)
### 日志
- [复现日志](./data/ml-1m/ENSFM.txt)
- [官方代码日志](./data/ml-1m/ENSFM-tf.txt)
## 5. TIPC测试
```
cd PaddleRec
bash test_tipc/prepare.sh ./test_tipc/configs/ensfm/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ensfm/train_infer_python.txt 'lite_train_lite_infer'
```