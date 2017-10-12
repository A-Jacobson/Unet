import tqdm
from tqdm import tqdm_notebook
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class Trainer:
    """

    """
    def __init__(self, model):
        self.model = model

    def fit_generator(self, dataset, criterion, optimizer, n_epochs=1, batch_size=1, shuffle=False):
        """
        :param dataset:
        :param criterion:
        :param optimizer:
        :param n_epochs:
        :param batch_size:
        :param shuffle:
        :return:
        """
        # TODO add validation
        loss_history = []
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for epoch in range(n_epochs):
            t = tqdm_notebook(enumerate(loader), total=len(loader))
            for batch, (data, target) in t:
                data, target = Variable(data.cuda()), Variable(target.cuda())
                output = self.model(data)
                loss = criterion(output, target)
                current_loss = loss.data[0]
                t.set_description("[ Epoch {} | Loss {:.4f} ] ".format(epoch, current_loss))
                loss_history.append(current_loss)
                loss.backward()
                optimizer.step()
        return loss_history

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = Variable(x.unsqueeze(0)).cuda()
        output = F.sigmoid(self.model(x)).data.cpu()
        y_pred = output
        return y_pred.numpy()

    def predict_generator(self, dataset, batch_size=1):
        predictions = []
        loader = DataLoader(dataset, batch_size=batch_size)
        for batch, (data, target) in tqdm_notebook(enumerate(loader), total=len(loader)):
            data = Variable(data.cuda())
            outputs = self.model(data)
            for prediction in outputs:
                predictions.append(prediction.data.cpu().numpy())
        return np.array(predictions)

    def evaluate(self, dataset, metrics):
        # TODO
        raise NotImplementedError
