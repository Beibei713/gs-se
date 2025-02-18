import torch
import torch.nn as nn

class SEWithAdvancedSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        改进后的通道和空间注意力结合的SE模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(SEWithAdvancedSpatialAttention, self).__init__()

        # 通道注意力部分
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)  # 降维
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid_channel = nn.Sigmoid()  # Sigmoid 激活生成通道权重

        # 空间注意力部分 (改进)
        self.conv3x3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)  # 3x3卷积
        self.conv5x5 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2, bias=False)  # 5x5卷积
        self.conv7x7 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)  # 7x7卷积
        self.conv_fuse = nn.Conv2d(3, 1, kernel_size=1, bias=False)  # 将多个尺度的空间信息进行融合
        self.sigmoid_spatial = nn.Sigmoid()  # Sigmoid 激活生成空间权重

    def forward(self, x):
        # 通道注意力部分
        if x.dim() == 4:  # 四维输入 [N, C, H, W]
            n, c, h, w = x.size()
            x_reshaped = x
        elif x.dim() == 5:  # 五维输入 [N, S, C, H, W]
            n, s, c, h, w = x.size()
            x_reshaped = x.view(n * s, c, h, w)
        else:
            raise ValueError("Input tensor must be 4D or 5D")

        y = self.global_avg_pool(x_reshaped)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y_channel = self.sigmoid_channel(y)  # 通道权重
        x_channel = x_reshaped * y_channel  # 按通道加权

        # 空间注意力部分（多尺度卷积）
        out_3x3 = self.conv3x3(x_channel)  # 使用3x3卷积提取空间特征
        out_5x5 = self.conv5x5(x_channel)  # 使用5x5卷积提取空间特征
        out_7x7 = self.conv7x7(x_channel)  # 使用7x7卷积提取空间特征

        # 融合多个尺度的空间特征
        out_fused = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)  # 拼接不同尺度的特征
        out_fused = self.conv_fuse(out_fused)  # 使用1x1卷积融合空间特征

        y_spatial = self.sigmoid_spatial(out_fused)  # 生成空间权重

        if x.dim() == 4:
            x_spatial = x_channel * y_spatial
        elif x.dim() == 5:
            x_spatial = (x_channel * y_spatial).view(n, s, c, h, w)

        return x_spatial  # 同时加权通道和空间





if __name__ == "__main__":
    # 输入张量，形状为 [batch_size, channels, height, width]
    input_tensor = torch.randn(8, 30, 64, 64, 44)  # 批量大小8，通道数64，特征图尺寸32x32
    se_block = SEWithAdvancedSpatialAttention(in_channels=64, reduction=16)
    output_tensor = se_block(input_tensor)  # 前向传播
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
