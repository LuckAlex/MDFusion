import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from collections import OrderedDict
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DCTBase(nn.Module):
    def __init__(self, N=8):
        super(DCTBase, self).__init__()
        self.N = N

    def mk_coff(self, N=8, rearrange=True):
        dct_weight = np.zeros((N * N, N, N))
        for k in range(N * N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N=N)  # from low frequency to high frequency
        return out_weight  # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N=8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy()  # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N - 1) and j % 2 == 0:
                j += 1
            elif (j == 0 or j == N - 1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k + 1, ...] = src_weight[index, ...]
        return rearrange_weigth

    def create_ycbcr_transform(self, matrix):
        transform = nn.Conv2d(3, 3, 1, 1, bias=False)
        transform_matrix = torch.from_numpy(matrix).float().unsqueeze(2).unsqueeze(3)
        transform.weight.data = transform_matrix
        transform.weight.requires_grad = False
        return transform


class DCT(DCTBase):
    def __init__(self, N=8, in_channal=3):
        super(DCT, self).__init__(N)
        self.N = N  # default is 8 for JPEG
        self.fre_len = N * N
        self.in_channal = in_channal
        self.out_channal = N * N * in_channal
        # self.weight = torch.from_numpy(self.mk_coff(N = N)).float().unsqueeze(1)

        # 3 H W -> N*N  H/N  W/N
        self.dct_conv = nn.Conv2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)

        # 64 *1 * 8 * 8, from low frequency to high fre
        self.weight = torch.from_numpy(self.mk_coff(N=N)).float().unsqueeze(1)
        # self.dct_conv = nn.Conv2d(1, self.out_channal, N, N, bias=False)
        self.dct_conv.weight.data = torch.cat([self.weight] * self.in_channal, dim=0)  # 64 1 8 8
        self.dct_conv.weight.requires_grad = False
        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)

        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])  # a similar version, maybe be a little wrong
        self.Ycbcr = self.create_ycbcr_transform(trans_matrix)

    def forward(self, x):
        dct = self.dct_conv(self.Ycbcr(x))
        return dct


class ReDCT(DCTBase):
    def __init__(self, N=8, in_channal=3):
        super(ReDCT, self).__init__(N)

        self.N = N  # default is 8 for JPEG
        self.in_channal = in_channal * N * N
        self.out_channal = in_channal
        self.fre_len = N * N

        self.weight = torch.from_numpy(self.mk_coff(N=N)).float().unsqueeze(1)

        self.reDCT = nn.ConvTranspose2d(self.in_channal, self.out_channal, self.N, self.N, bias=False,
                                        groups=self.out_channal)
        self.reDCT.weight.data = torch.cat([self.weight] * self.out_channal, dim=0)
        self.reDCT.weight.requires_grad = False
        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)

        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                             [-0.169, -0.331, 0.5],
                                             [0.5, -0.419, -0.081]]))
        self.reYcbcr = self.create_ycbcr_transform(re_matrix)

    def forward(self, dct):
        out = self.reDCT(dct)
        out = self.reYcbcr(out)
        return out


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.conv1 = nn.Conv2d(192, 64, kernel_size=1)  # Convert 192 channels to 64 channels
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = self.conv1

        # Create the FPN using the feature maps from ResNet
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=192
        )

        # Initialize adaptive average pooling layers for each FPN output
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((32, 32))

        # Initialize batch normalization and ReLU layers
        # self.bn = nn.BatchNorm2d(192)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Pass the input through the ResNet backbone to get feature maps
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x1 = self.resnet50.layer1(x)
        x2 = self.resnet50.layer2(x1)
        x3 = self.resnet50.layer3(x2)
        x4 = self.resnet50.layer4(x3)

        # Pass the feature maps through the FPN
        input_to_fpn = OrderedDict([('0', x1), ('1', x2), ('2', x3), ('3', x4)])
        fpn_out = self.fpn(input_to_fpn)

        # Apply adaptive average pooling, batch normalization and ReLU to each FPN output
        pooled_outputs = []
        for key in fpn_out:
            pooled_output = self.adaptive_avg_pool(fpn_out[key])
            # pooled_output = self.bn(pooled_output)
            # pooled_output = self.relu(pooled_output)
            # pooled_outputs.append(torch.tanh(pooled_output))
            pooled_outputs.append(pooled_output)
        return pooled_outputs


class CustomModel(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(CustomModel, self).__init__()
        self.dct = DCT()
        self.fpn = FPN()
        self.rdct = ReDCT()


    def forward(self, x):
        # Apply DCT transformation to the input
        x_dct = self.dct(x)
        x_dct_ori = x_dct.clone()

        # Pass the transformed input through the FPN layers
        att_noise = self.fpn(x_dct)

        # Merge the input and watermark using the watermark embedding layer
        mix_output = x_dct_ori + (att_noise[0] + att_noise[1] + att_noise[2] + att_noise[3])/ 40
        # mix_output = x_dct_ori

        output_tensor = self.rdct(mix_output)
        output = torch.clamp(output_tensor, min=0, max=1)
        # output = (torch.tanh(output_tensor) + 1) / 2

        return output



if __name__ == '__main__':

    image_size = 256

    bs = 1

    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.520, 0.425, 0.380], std=[0.253, 0.228, 0.225])
    ])

    dataset = datasets.ImageFolder(root='./test_data',
                                   transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=bs, shuffle=True,
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 drop_last=True)

    model = CustomModel()
    model.to(device)
    img_list = []

    for i, data in enumerate(dataset_loader, 0):
        # get the inputs
        inputs, _ = data
        input_tensor = inputs.to(device)




        output = model(input_tensor)

        plt.subplot(211)
        plt.imshow(input_tensor.detach().cpu().numpy()[0].transpose([1, 2, 0]))

        plt.subplot(212)
        plt.imshow(output.detach().cpu().numpy()[0].transpose([1, 2, 0]))

        plt.show()





