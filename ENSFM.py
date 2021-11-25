import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
import time
import sys
import argparse

import tqdm

import LoadData as DATA
import random

default_type = paddle.get_default_dtype()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ENSFM")
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=501,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=1,
                        help='dropout keep_prob')
    parser.add_argument('--negative_weight', type=float, default=0.5,
                        help='weight of non-observed data')
    parser.add_argument('--topK', nargs='?', type=int, default=[5, 10, 20],
                        help='topK for hr/ndcg')
    parser.add_argument('--seed', type=int, default=2019, )
    return parser.parse_args()


def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


class ENSFM(nn.Layer):
    def __init__(self, item_attribute, user_field_M, item_field_M, embedding_size, max_item_pu, args):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_item_pu = max_item_pu
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.weight1 = args.negative_weight
        self.item_attribute = paddle.to_tensor(item_attribute)

        self.user_feature_emb = nn.Embedding(self.user_field_M, self.embedding_size,
                                             weight_attr=paddle.ParamAttr(
                                                 initializer=paddle.nn.initializer.TruncatedNormal(std=0.01)))
        self.H_i = paddle.create_parameter([self.embedding_size, 1], default_type,
                                           default_initializer=nn.initializer.Constant(0.01))
        self.H_s = paddle.create_parameter([self.embedding_size, 1], default_type,
                                           default_initializer=nn.initializer.Constant(0.01))
        self.all_item_feature_emb = nn.Embedding(self.item_field_M + 1, self.embedding_size,
                                                 weight_attr=paddle.ParamAttr(
                                                     initializer=paddle.nn.initializer.TruncatedNormal(std=0.01)))

        self.user_bias = nn.Embedding(self.user_field_M, 1,
                                      weight_attr=paddle.ParamAttr(
                                          initializer=paddle.nn.initializer.TruncatedNormal(std=0.01)))
        self.item_bias = nn.Embedding(self.item_field_M, 1,
                                      weight_attr=paddle.ParamAttr(
                                          initializer=paddle.nn.initializer.TruncatedNormal(std=0.01)))
        self.bias = paddle.create_parameter([1], default_type,
                                            default_initializer=nn.initializer.Constant(0.))

    def forward(self, input_u, input_ur=None):
        user_feature_emb = self.user_feature_emb(input_u)
        summed_user_emb = user_feature_emb.sum(1)
        all_item_feature_emb = self.all_item_feature_emb(self.item_attribute)
        summed_all_item_emb = all_item_feature_emb.sum(1)
        user_cross = 0.5 * (summed_user_emb ** 2 - (user_feature_emb ** 2).sum(1))
        item_cross = 0.5 * (summed_all_item_emb ** 2 - (all_item_feature_emb ** 2).sum(1))
        user_cross_score = user_cross.matmul(self.H_s)
        item_cross_score = item_cross.matmul(self.H_s)
        user_bias = self.user_bias(input_u).sum(1)
        item_bias = self.item_bias(self.item_attribute).sum(1)

        I = paddle.ones([input_u.shape[0], 1])
        p_emb = paddle.concat([summed_user_emb, user_cross_score + user_bias + self.bias, I], 1)

        I = paddle.ones([summed_all_item_emb.shape[0], 1])
        q_emb = paddle.concat([summed_all_item_emb, I, item_cross_score + item_bias], 1)
        H_i_emb = paddle.concat([self.H_i, paddle.to_tensor([[1.0]]), paddle.to_tensor([[1.0]])], 0)
        dot = paddle.einsum('ac,bc->abc', p_emb, q_emb)
        pre = paddle.einsum('ajk,kl->aj', dot, H_i_emb)
        if input_ur is None:
            return (pre,)

        pos_item = F.embedding(input_ur, q_emb)
        pos_num_r = (input_ur != data.item_bind_M).astype(default_type)
        pos_item = paddle.einsum('ab,abc->abc', pos_num_r, pos_item)

        pos_r = paddle.einsum('ac,abc->abc', p_emb, pos_item)
        pos_r = paddle.einsum('ajk,kl->ajl', pos_r, H_i_emb).reshape([-1, self.max_item_pu])
        return pre, pos_r, q_emb, p_emb, H_i_emb


@paddle.no_grad()
def evaluate():
    model.eval()
    eva_batch = 128
    recall50 = []
    recall100 = []
    recall200 = []
    ndcg50 = []
    ndcg100 = []
    ndcg200 = []

    user_features = data.user_test
    ll = int(len(user_features) / eva_batch) + 1
    for batch_num in range(ll):
        start_index = batch_num * eva_batch
        end_index = min((batch_num + 1) * eva_batch, len(user_features))
        u_batch = paddle.to_tensor(user_features[start_index:end_index])
        batch_users = end_index - start_index
        with paddle.no_grad():
            pre, *_ = model(u_batch)
        pre = pre.cpu().numpy()
        pre = np.delete(pre, -1, axis=1)

        user_id = []
        for one in u_batch.cpu().numpy():
            user_id.append(data.binded_users["-".join([str((item)) for item in one[0:]])])

        idx = np.zeros_like(pre, dtype=bool)
        idx[data.Train_data[user_id].nonzero()] = True
        pre[idx] = -np.inf

        # recall

        recall = []

        for kj in args.topK:
            idx_topk_part = np.argpartition(-pre, kj, 1)

            pre_bin = np.zeros_like(pre, dtype=bool)
            pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]] = True

            true_bin = np.zeros_like(pre, dtype=bool)
            true_bin[data.Test_data[user_id].nonzero()] = True

            tmp = (np.logical_and(true_bin, pre_bin).sum(axis=1)).astype(np.float32)
            recall.append(tmp / np.minimum(kj, true_bin.sum(axis=1)))

        # ndcg
        ndcg = []

        for kj in args.topK:
            idx_topk_part = np.argpartition(-pre, kj, 1)

            topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]]
            idx_part = np.argsort(-topk_part, axis=1)
            idx_topk = idx_topk_part[np.arange(end_index - start_index)[:, np.newaxis], idx_part]

            tp = np.log(2) / np.log(np.arange(2, kj + 2))

            test_batch = data.Test_data[user_id]
            DCG = (test_batch[np.arange(batch_users)[:, np.newaxis],
                              idx_topk].toarray() * tp).sum(axis=1)

            IDCG = np.array([(tp[:min(n, kj)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            ndcg.append(DCG / IDCG)

        recall50.append(recall[0])
        recall100.append(recall[1])
        recall200.append(recall[2])
        ndcg50.append(ndcg[0])
        ndcg100.append(ndcg[1])
        ndcg200.append(ndcg[2])

    recall50 = np.hstack(recall50)
    recall100 = np.hstack(recall100)
    recall200 = np.hstack(recall200)
    ndcg50 = np.hstack(ndcg50)
    ndcg100 = np.hstack(ndcg100)
    ndcg200 = np.hstack(ndcg200)

    write_str = '\t'.join(map(str, [round(np.mean(recall50), 4),
                                    round(np.mean(recall100), 4),
                                    round(np.mean(recall200), 4)])) + '\n'
    f1.write(write_str)
    f1.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


if __name__ == '__main__':

    args = parse_args()
    set_seed(args.seed)
    if args.dataset == 'laspaddlem':
        print('load laspaddlem data')
        DATA_ROOT = './data/laspaddlem'
    if args.dataset == 'frappe':
        print('load frappe data')
        DATA_ROOT = './data/frappe'
    if args.dataset == 'ml-1m':
        print('load ml-1m data')
        DATA_ROOT = './data/ml-1m'

    f1 = open(os.path.join(DATA_ROOT, 'ENSFM.txt'), 'w')
    data = DATA.LoadData(DATA_ROOT)
    model = ENSFM(data.item_map_list, data.user_field_M, data.item_field_M, args.embed_size, data.max_positive_len,
                  args)

    batch_size = args.batch_size
    optimizer = paddle.optimizer.Adagrad(learning_rate=args.lr, initial_accumulator_value=1e-8,
                                         parameters=model.parameters())
    # optimizer = paddle.optimizer.Adam(learning_rate=1e-3,parameters=model.parameters())
    for epoch in tqdm.tqdm(range(args.epochs)):
        shuffle_indices = np.random.permutation(np.arange(len(data.user_train)))
        data.user_train = data.user_train[shuffle_indices]
        data.item_train = data.item_train[shuffle_indices]

        ll = int(len(data.user_train) / batch_size)
        total_loss = 0

        for batch_num in range(ll):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(data.user_train))
            u_batch = paddle.to_tensor(data.user_train[start_index:end_index])
            i_batch = paddle.to_tensor(data.item_train[start_index:end_index])
            pre, pos_r, q_emb, p_emb, H_i_emb = model(u_batch, i_batch)

            loss = args.negative_weight * paddle.sum(
                paddle.sum(paddle.sum(paddle.einsum('ab,ac->abc', q_emb, q_emb), 0)
                           * paddle.sum(paddle.einsum('ab,ac->abc', p_emb, p_emb), 0)
                           * paddle.matmul(H_i_emb, H_i_emb, transpose_y=True), 0), 0)
            loss += paddle.sum((1.0 - args.negative_weight) * paddle.square(pos_r) - 2.0 * pos_r)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        # print(total_loss / ll)
        if epoch < args.epochs:
            if epoch % args.verbose == 0:
                evaluate()

        if epoch >= args.epochs:
            evaluate()
