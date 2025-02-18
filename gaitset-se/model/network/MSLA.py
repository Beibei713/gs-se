import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):


        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)



        return x / x_exp_pool

class AttentionPropagationLocalAttention(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, f, 1)
        self.softpool = SoftPooling2D(kernel_size=7, stride=3, padding=1)  # 
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(f, channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        # 新增注意力传播卷积层
        self.propagate_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        if x.dim() == 4:  # 四维输入 [N, C, H, W]
            n, c, h, w = x.size()
            x_reshaped = x
        elif x.dim() == 5:  # 五维输入 [N, S, C, H, W]
            n, s, c, h, w = x.size()
            x_reshaped = x.view(n * s, c, h, w)
        else:
            raise ValueError("Input tensor must be 4D or 5D")

        # 计算局部重要性
        x1 = self.conv1(x_reshaped)
        x2 = self.softpool(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        y = self.sigmoid(x4)

        # 调整 y 的尺寸使其与 x 匹配
        y = F.interpolate(y, size=(x_reshaped.size(2), x_reshaped.size(3)), mode='bilinear', align_corners=False)

        # 注意力传播
        x_propagated = self.propagate_conv(x_reshaped * y)


        # 门控机制
        g = self.sigmoid(x_reshaped[:, :1, :, :].clone())  # 确保 g 的维度与 x 匹配

        if x.dim() == 4:
            x_spatial = (x_reshaped * y * g + x_propagated)
        elif x.dim() == 5:
            x_spatialL = (x_reshaped * y * g + x_propagated)
            x_spatial = x_spatialL.view(n, s, c, h, w)

        return x_spatial

if __name__ == "__main__":
    input = torch.randn(30, 32, 64, 44)
    ALA = AttentionPropagationLocalAttention(32)
    output = ALA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
