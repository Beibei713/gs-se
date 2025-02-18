import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda() # 将NumPy数组转换为Torch张量，并传送到GPU
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    # 计算x和y的平方和，并进行扩展以便进行矩阵运算
    dist = torch.sqrt(F.relu(dist))  # 计算距离的平方根，并确保距离非负
    return dist


def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]  # 从配置中获取数据集名称
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)
    # 定义不同数据集的探测序列和画廊序列
    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}


    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                  ##  print("Distance matrix sample:", dist)
                    idx = dist.sort(1)[1].cpu().numpy()
                  ##  print("Sorted index sample:", idx)

                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

                    test_acc = np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                      0) * 100 / dist.shape[0]
                  ##  print("Test accuracy calculation:", test_acc)

    return acc
