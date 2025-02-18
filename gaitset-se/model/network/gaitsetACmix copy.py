import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .basic_blocks import SetBlock, BasicConv2d
from .acmix import ACmix
from .SEWithSpatialAttention import SEWithAdvancedSpatialAttention
class SetNetWithACmix(nn.Module):


    def __init__(self, hidden_dim):
        super(SetNetWithACmix, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))


        _gl_in_channels = 32
        _gl_channels = [64, 128]
        self.gl_layer1 = BasicConv2d(_gl_in_channels, _gl_channels[0], 3, padding=1)
        self.gl_layer2 = BasicConv2d(_gl_channels[0], _gl_channels[0], 3, padding=1)
        self.gl_layer3 = BasicConv2d(_gl_channels[0], _gl_channels[1], 3, padding=1)
        self.gl_layer4 = BasicConv2d(_gl_channels[1], _gl_channels[1], 3, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)


        self.se_block = SEWithAdvancedSpatialAttention(in_channels=128, reduction=16)

                # 归一化
        self.bn_fg = nn.BatchNorm2d(_set_channels[2])
        self.bn_x = nn.BatchNorm2d(_set_channels[2])
        self.bn_gl = nn.BatchNorm2d(_gl_channels[1])


        self.bin_num = [1, 2, 4, 8, 16]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num) * 3, 128, hidden_dim)))])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

            # for name, param in self.named_parameters():
            #  if param.requires_grad:
            #   print(f"{name}: Mean = {param.data.mean().item()}, Std = {param.data.std().item()}")

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list



    def forward(self, silho, batch_frame=None):
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]

            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()


        n = silho.size(0)
        x = silho.unsqueeze(2)
        fg = silho.unsqueeze(2)
        fg_top = fg[:,:,:, :12, :]  # 第一层 10 像素高
        fg_middle = fg[:,:,:, 12:40, :]  # 第二层 10-26 像素高
        fg_bottom = fg[:,:,:, 40:, :]  # 其余为第三层

        del silho


        x = self.set_layer1(x)
        x = self.set_layer2(x)


        gl = self.gl_layer1(self.frame_max(x)[0])
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)
        x = self.set_layer3(x)
        x = self.set_layer4(x)





        gl = self.gl_layer3(gl + self.frame_max(x)[0])
        gl = self.gl_layer4(gl)
        x = self.set_layer5(x)
        x = self.set_layer6(x)




        x = self.frame_max(x)[0]
        x = self.bn_x(x)
        gl = self.bn_gl(gl)
        gl = gl + x 

        fg_top = self.set_layer1(fg_top)
        fg_top = self.set_layer2(fg_top)
        fg_top = self.set_layer3(fg_top)
        fg_top = self.set_layer4(fg_top)
        fg_top = self.set_layer5(fg_top)
        fg_top = self.set_layer6(fg_top)

        # 对 fg_middle 进行六层卷积
        fg_middle = self.set_layer1(fg_middle)
        fg_middle = self.set_layer2(fg_middle)
        fg_middle = self.set_layer3(fg_middle)
        fg_middle = self.set_layer4(fg_middle)
        fg_middle = self.set_layer5(fg_middle)
        fg_middle = self.set_layer6(fg_middle)

        # 对 fg_bottom 进行六层卷积
        fg_bottom = self.set_layer1(fg_bottom)
        fg_bottom = self.set_layer2(fg_bottom)
        fg_bottom = self.set_layer3(fg_bottom)
        fg_bottom = self.set_layer4(fg_bottom)
        fg_bottom = self.set_layer5(fg_bottom)
        fg_bottom = self.set_layer6(fg_bottom)



        fg_top = self.se_block(fg_top)
        fg_middle = self.se_block(fg_middle)
        fg_bottom = self.se_block(fg_bottom)



        fg = torch.cat((fg_top*0.5, fg_middle*3, fg_bottom*3), dim=3)

        fg = self.frame_max(fg)[0]
        fg = self.bn_fg(fg)


        # fg = self.AFE_layer(fg)
        # x = self.acmix_layer6(x)
        # gl = self.glac_layer4(gl)

        feature = list()
        n, c, h, w = gl.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = gl.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = fg.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()


        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None



