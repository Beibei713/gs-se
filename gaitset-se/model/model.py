import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
from torch.optim.lr_scheduler import StepLR  # 明确导入StepLR
from .network import TripletLoss, SetNetWithACmix 
from .utils import TripletSampler


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source


        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip  # 默认：full
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size
        self.restore_iter = restore_iter
        self.total_iter = total_iter  # 迭代次数
        self.img_size = img_size  # 64

        # 初始化网络.
        self.encoder = SetNetWithACmix(self.hidden_dim).float()
        self.encoder = nn.DataParallel(self.encoder)  # 使用DataParallel进行多GPU训练
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)  # 使用DataParallel进行多GPU训练
        # 将模型和损失函数移动到GPU上
        self.encoder.cuda()
        self.triplet_loss.cuda()
        # 设置优化器
        #self.optimizer = optim.Adam([
        #    {'params': self.encoder.parameters()},
        #], lr=self.lr)
        self.optimizer = optim.SGD([
            {'params': self.encoder.parameters()},
        ], lr=self.lr, momentum=0.95)
        
        # 设置学习率调度器
        self.scheduler = StepLR(self.optimizer, step_size=20000, gamma=0.1)



        # 初始化一些用于跟踪训练过程的列表
        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []

        self.dist_list = []
        self.mean_dist = 0.01
        self.sample_type = 'all'

    def collate_fn(self, batch):  ##collate_fn 是一个自定义的批处理函数，用于在 DataLoader 中收集样本。这个函数定义了如何合并多个样本成一个批次。
        # 提取批次中的各个部分（序列、视图、序列类型、标签等）
        # batch是一个list 大小是128，每一个list有5维 (frame*64*44,数字 0-frame,角度,bg-02,id),应该是调用for in trainloder的时候才会执行这个地方，生成规定的格式
        batch_size = len(batch)
        feature_num = len(batch[0][0])  ##每个样本中包含的特征数。
        seqs = [batch[i][0] for i in range(batch_size)]  ##从批次中提取的序列部分
        frame_sets = [batch[i][1] for i in range(batch_size)]  ##从批次中提取的帧集合部分
        view = [batch[i][2] for i in range(batch_size)]  ##从批次中提取的视角部分
        seq_type = [batch[i][3] for i in range(batch_size)]  ##从批次中提取的序列类型部分
        label = [batch[i][4] for i in range(batch_size)]  ##从批次中提取的标签部分
        batch = [seqs, view, seq_type, label, None]  ## 重新组合成包含序列、视角、序列类型和标签的列表

        # 根据 `sample_type` 选择帧
        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            # 如果 `sample_type` 是 'random'，随机选择帧
            # 否则，选择所有帧，并可能根据GPU数量进行分割和填充
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                # 这里的random.choices是有放回的抽取样本,k是选取次数，这里的frame_num=30
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                len(frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
            seqs = [np.asarray([
                np.pad(seqs[j][_],
                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                       'constant',
                       constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):  ##训练  加载权重
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()  # 对于有dropout和BathNorm的训练要 .train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers, )


        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()  ##对标签排序  # 里面没有005,73个id 进行排序

        _time1 = datetime.now()  # 计时
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()  # 梯度清零

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, label_prob = self.encoder(*seq, batch_frame)
            # feature的维度是torch.Size([128, 62, 256])，label_prob=None 62的由来，两个31维度的特征concat成62维度

            target_label = [train_label_set.index(l) for l in label]  # list.index() 返回索引位置，每个label在label_set中的索引位置
            target_label = self.np2var(np.array(target_label)).long()
            # 训练标签转换为tensor torch.Size([128])  target_label.size:(128),label变成了索引位置
            # print(target_label.size())



            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num) = self.triplet_loss(triplet_feature,
                                                                                               triplet_label)

            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())  
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())  
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())  
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())


            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            if self.restore_iter % 50 == 0:
                print('iter {}:'.format(self.restore_iter), end='')
                print("50次训练时间", datetime.now() - _time1)
                log_data = [
                    'iter {}:'.format(self.restore_iter),
                    ', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)),
                    ', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)),
                    ', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)),

                    ', mean_dist={0:.8f}'.format(np.mean(self.dist_list)),
                    ', loss={0:.8f}'.format(loss.mean().data.cpu().numpy()),  # 添加这一行
                    ', lr=%f' % self.optimizer.param_groups[0]['lr'],
                    ', hard or full=%r' % self.hard_or_full_trip
                ]
                # 将数据拼接成一个完整的字符串
                log_message = ''.join(log_data)

                checkpoint_dir = osp.join('checkpoint', self.model_name)
                os.makedirs(checkpoint_dir, exist_ok=True)  # 如果目录不存在，创建目录
                log_file_path = osp.join(checkpoint_dir, 'training_log.txt')  # 日志文件路径

                sys.stdout.flush()
                # 写入到txt文件
                with open(log_file_path, 'a') as log_file:  # 使用'a'模式打开文件，以追加内容
                    log_file.write(log_message + '\n')  # 写入数据后换行

                _time1 = datetime.now()

            if self.restore_iter % 2000 == 0:  # 50次迭代打印
                self.save()  # 每训练50次，保存一次模型
                # 准备要写入文件的数据

                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []

                self.dist_list = []

            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):  # 测试
        self.encoder.eval()
        source = self.test_source \
            if flag == 'test' \
            else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            ## print(f"Batch {i}, seq: {seq}, view: {view}, seq_type: {seq_type}, label: {label}")

            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            ##print(batch_frame, np.sum(batch_frame))

            feature, _ = self.encoder(*seq, batch_frame)
            ##print(f"Feature for batch {i}: {feature}")

            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load   #restore_iter：要加载的检查点的迭代索引

    def load(self, restore_iter):
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
