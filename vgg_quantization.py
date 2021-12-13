import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.quantization import QuantStub, DeQuantStub
from torchvision import datasets, transforms
import warnings

warnings.filterwarnings('ignore')


cuda = False
test_batch_size = 32


def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    model.eval()
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def cal_scale_and_zeros(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    model.eval()
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        model(data)


class vggQ(nn.Module):
    def __init__(self):
        super(vggQ, self).__init__()
        self.feature1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.down1 = nn.MaxPool2d(2)
        self.feature2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.down2 = nn.MaxPool2d(2)
        self.feature3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.down3 = nn.MaxPool2d(2)
        self.adp = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.feature1(x)
        x = self.down1(x)
        x = self.feature2(x)
        x = self.down2(x)
        x = self.feature3(x)
        x = self.down3(x)
        x = self.adp(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        y = self.dequant(y)
        return y

    def fuse_model(self):
        torch.quantization.fuse_modules(self.feature1, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.feature2, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.feature3, ['0', '1', '2'], inplace=True)


model = vggQ()
model.load_state_dict(torch.load('./model_best.pth.tar')['state_dict'], strict=True)
model.eval()
from torch.utils.mobile_optimizer import optimize_for_mobile
model.eval()
example = torch.rand(1, 3, 32, 32)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("./model_fp32.pt")
t1 = time.time()
test(model)
print("float 32 inference time:", time.time() - t1)


model.float()
model.eval()
# 算子融合，可以减小量化误差提升精度
model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32_prepared = torch.quantization.prepare(model, inplace=True)
# cal scale and zeros，少量的数据校准，不校准计算scale和zeros的精度在13%左右，校准后可以提升到80.9%
cal_scale_and_zeros(model)
model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=True)
torch.save(model_int8.state_dict(), "./model_int8.pth")


t1 = time.time()
test(model_int8)
print("int 8 inference time:", time.time() - t1)


from torch.utils.mobile_optimizer import optimize_for_mobile
model_int8.eval()
example = torch.rand(1, 3, 32, 32)
traced_script_module = torch.jit.trace(model_int8, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("./model_int8.pt")





