import torch
import torchvision
from torchsummary import summary


if __name__ == '__main__':
    m = torchvision.models.resnet50(pretrained=True, progress=True)
    m = m.cuda()
    # for p in m.parameters():
    #     print(p.numel())
    #     print(p.shape)
    #     print(torch.count_nonzero(torch.abs(p) >= 50.))
    #     print(torch.count_nonzero(torch.abs(p) >= 10.))
    #     print(torch.count_nonzero(torch.abs(p) >= 1.))
    #     print(torch.count_nonzero(torch.abs(p) >= 0.1))

    summary(m, (3, 224, 224))