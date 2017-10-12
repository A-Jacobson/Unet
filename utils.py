from torch.autograd import Variable
import torch


def to_variable(arr, cuda=True):
    return Variable(torch.from_numpy(arr.reshape(1, 1, 512, 512)).float()).cuda()


def to_numpy(variable):
    return variable.data.cpu().numpy()
