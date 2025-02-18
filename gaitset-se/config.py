conf = {
    "WORK_PATH": "/beibei/gaitSet/GaitSet-ACmix/work",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "data": {
        #'dataset_path': "/beibei/CASIA-B-processed",
        'dataset_path': "/beibei/CASIA-B-processed",
        'resolution': '64',##图像分辨率
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,  #*在124个中随机选出73个人
    },
    "model": {
        'hidden_dim': 256,##模型的隐藏层维度
        'lr': 1e-2,#*学习率为0.0001
        'hard_or_full_trip': 'full',##三元组选择策略
        'batch_size': (8,16),#*批次p*k = 8*16,其中p是人数，k是p个人每人拿k个样本
        'restore_iter': 0,#恢复训练时的迭代次数，设为 0 表示从头开始训练。
        'total_iter': 80000,##总迭代次数，设为 80000。
        'margin': 0.2,##三元组损失中的边距值
        'num_workers': 1,##数据加载时使用的工作线程数
        'frame_num': 30,##每个序列的帧数
        'model_name': 'GaitSet-18',##模型名称   \gaitset-ACmix
    },
}
