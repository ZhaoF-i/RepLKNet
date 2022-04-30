import torch.nn as nn
import torch.autograd.variable
from torch.autograd.variable import *
from utils.stft_istft_real_imag_hamming import STFT
from utils.FrameOptions import frames_data, frames_overlap, pad_input
from thop import profile, clever_format

class CRN_Stage1(nn.Module):
    def __init__(self,win_len,win_offset):
        self.win_len = win_len
        self.win_offset = win_offset
        super(CRN_Stage1, self).__init__()
        self.lstm_input_size = 64 * 9
        self.lstm_layers = 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 3), stride=(1, 2))
        self.conv1_relu = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 3), stride=(1, 2))
        self.conv2_relu = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2))
        self.conv3_relu = nn.ELU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv4_relu = nn.ELU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv5_relu = nn.ELU()
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_input_size,
                            num_layers=self.lstm_layers,
                            batch_first=True)

        self.conv5_t = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv5_t_relu = nn.ELU()
        self.conv4_t = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv4_t_relu = nn.ELU()
        self.conv3_t = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv3_t_relu = nn.ELU()
        self.conv2_t = nn.ConvTranspose2d(in_channels=16 * 2, out_channels=8, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv2_t_relu = nn.ELU()
        self.conv1_t = nn.ConvTranspose2d(in_channels=8 * 2, out_channels=1, kernel_size=(2, 3), stride=(1, 2),
                                          output_padding=(0, 1), padding=(1, 0))
        self.conv1_t_relu = nn.Softplus()

        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5_bn = nn.BatchNorm2d(64)

        self.conv5_t_bn = nn.BatchNorm2d(64)
        self.conv4_t_bn = nn.BatchNorm2d(32)
        self.conv3_t_bn = nn.BatchNorm2d(16)
        self.conv2_t_bn = nn.BatchNorm2d(8)
        self.conv1_t_bn = nn.BatchNorm2d(1)
        self.pad = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.STFT = STFT(self.win_len, self.win_offset).cuda()

    def forward(self, input_data_c1):

        input_feature = frames_data(input_data_c1, self.win_offset, True)

        e1 = self.conv1_relu(self.conv1_bn(self.conv1(self.pad(torch.stack([input_feature], 1)))))
        e2 = self.conv2_relu(self.conv2_bn(self.conv2(self.pad(e1))))
        e3 = self.conv3_relu(self.conv3_bn(self.conv3(self.pad(e2))))
        e4 = self.conv4_relu(self.conv4_bn(self.conv4(self.pad(e3))))
        e5 = self.conv5_relu(self.conv5_bn(self.conv5(self.pad(e4))))

        self.lstm.flatten_parameters()
        out_real = e5.contiguous().transpose(1, 2)
        out_real = out_real.contiguous().view(out_real.size(0), out_real.size(1), -1)
        lstm_out, _ = self.lstm(out_real)
        lstm_out_real = lstm_out.contiguous().view(lstm_out.size(0), lstm_out.size(1), 64, 9)
        lstm_out_real = lstm_out_real.contiguous().transpose(1, 2)

        t5 = self.conv5_t_relu(self.conv5_t_bn(self.conv5_t(self.pad(torch.cat((lstm_out_real, e5), dim=1)))))
        t4 = self.conv4_t_relu(self.conv4_t_bn(self.conv4_t(self.pad(torch.cat((t5, e4), dim=1)))))
        t3 = self.conv3_t_relu(self.conv3_t_bn(self.conv3_t(self.pad(torch.cat((t4, e3), dim=1)))))
        t2 = self.conv2_t_relu(self.conv2_t_bn(self.conv2_t(self.pad(torch.cat((t3, e2), dim=1)))))
        t1 = self.conv1_t_bn(self.conv1_t(self.pad(torch.cat((t2, e1), dim=1))))
        out = torch.squeeze(t1, 1)
        out = frames_overlap(out, self.win_offset, True)
        return out

class CRN_Stage2(nn.Module):
    def __init__(self,win_len,win_offset):
        self.win_len = win_len
        self.win_offset = win_offset
        super(CRN_Stage2, self).__init__()
        self.lstm_input_size = 128 * 4
        self.lstm_layers = 2
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 3), stride=(1, 2))
        self.conv1_relu = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 3), stride=(1, 2))
        self.conv2_relu = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2))
        self.conv3_relu = nn.ELU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv4_relu = nn.ELU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2))
        self.conv5_relu = nn.ELU()
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_input_size,
                            num_layers=self.lstm_layers,
                            batch_first=True)

        self.conv5_t = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv5_t_relu = nn.ELU()
        self.conv4_t = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv4_t_relu = nn.ELU()
        self.conv3_t = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv3_t_relu = nn.ELU()
        self.conv2_t = nn.ConvTranspose2d(in_channels=16 * 2, out_channels=8, kernel_size=(2, 3), stride=(1, 2),
                                          output_padding=(0, 1), padding=(1, 0))
        self.conv2_t_relu = nn.ELU()
        self.conv1_t = nn.ConvTranspose2d(in_channels=8 * 2, out_channels=2, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv1_t_relu = nn.Softplus()

        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5_bn = nn.BatchNorm2d(128)

        self.conv5_t_bn = nn.BatchNorm2d(64)
        self.conv4_t_bn = nn.BatchNorm2d(32)
        self.conv3_t_bn = nn.BatchNorm2d(16)
        self.conv2_t_bn = nn.BatchNorm2d(8)
        self.conv1_t_bn = nn.BatchNorm2d(2)
        self.pad = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        # self.pad_t = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.STFT = STFT(self.win_len, self.win_offset).cuda()
        self.linear_real = nn.Linear(161, 161)
        self.linear_imag = nn.Linear(161, 161)

    def forward(self, mix, nestFrameNoise):
        # input_data_c1 = train_info_.mix_feat_b
        STFT_C1 = self.STFT.transform(mix)
        input_feature_mix = STFT_C1.permute(0, 3, 1, 2)


        STFT_C2 = self.STFT.transform(nestFrameNoise).permute(0, 3, 1, 2)
        input_feature_noise = self.pad(STFT_C2)[:, :, :-1, :]

        input_feature = torch.cat((input_feature_mix, input_feature_noise), dim=1)

        e1 = self.conv1_relu(self.conv1_bn(self.conv1(self.pad(input_feature))))
        e2 = self.conv2_relu(self.conv2_bn(self.conv2(self.pad(e1))))
        e3 = self.conv3_relu(self.conv3_bn(self.conv3(self.pad(e2))))
        e4 = self.conv4_relu(self.conv4_bn(self.conv4(self.pad(e3))))
        e5 = self.conv5_relu(self.conv5_bn(self.conv5(self.pad(e4))))

        self.lstm.flatten_parameters()
        out_real = e5.contiguous().transpose(1, 2)
        out_real = out_real.contiguous().view(out_real.size(0), out_real.size(1), -1)
        lstm_out, _ = self.lstm(out_real)
        lstm_out_real = lstm_out.contiguous().view(lstm_out.size(0), lstm_out.size(1), 128, 4)
        lstm_out_real = lstm_out_real.contiguous().transpose(1, 2)

        t5 = self.conv5_t_relu(self.conv5_t_bn(self.conv5_t(self.pad(torch.cat((lstm_out_real, e5), dim=1)))))
        t4 = self.conv4_t_relu(self.conv4_t_bn(self.conv4_t(self.pad(torch.cat((t5, e4), dim=1)))))
        t3 = self.conv3_t_relu(self.conv3_t_bn(self.conv3_t(self.pad(torch.cat((t4, e3), dim=1)))))
        t2 = self.conv2_t_relu(self.conv2_t_bn(self.conv2_t(self.pad(torch.cat((t3, e2), dim=1)))))
        t1 = self.conv1_t_relu(self.conv1_t_bn(self.conv1_t(self.pad(torch.cat((t2, e1), dim=1)))))
        out_r = self.linear_real(t1[:, 0, :, :])
        out_i = self.linear_imag(t1[:, 1, :, :])
        out = self.STFT.inverse(torch.stack([out_r, out_i], dim=1).permute(0, 2, 3, 1))
        return out

class NET_Wrapper(nn.Module):
    def __init__(self, win_len, win_offset):
        self.win_len = win_len
        self.win_offset = win_offset
        super(NET_Wrapper, self).__init__()
        self.stage1 = CRN_Stage1(self.win_len, self.win_offset)
        self.stage2 = CRN_Stage2(self.win_len, self.win_offset)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(input, stage1_out)
        return [stage1_out, stage2_out]


if __name__ == '__main__':
    input = Variable(torch.FloatTensor(torch.rand(1, 16000))).cuda(0)
    net = NET_Wrapper(320, 160).cuda()
    macs, params = profile(net, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
    # print("%s | %.2f | %.2f" % ('elephantstudent', params, macs))