# 模型名称 Efficient Non-Sampling Factorization Machines for Optimal Context-Aware Recommendation
## 1. 简介
本论文提出从全部数据中学习 FM 来进行 Top-N 推荐，并设计了一个高效的非采样分解机框架（Efficient Non-Sampling Factorization Machines, ENSFM）。通过严格的数学推导，ENSFM 不仅在两类常用的推荐方法——分解机（FM）和矩阵分解（MF）之间建造了一个桥梁，并且可以高效的从整体数据中学习 FM 参数。 
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