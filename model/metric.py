import torch
import sklearn.metrics


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def log_loss(output, target):
    output = output.cpu()
    target = target.cpu()
    with torch.no_grad():
        return sklearn.metrics.log_loss(target.cpu().numpy(), output.numpy())


def roc_auc_score(output, target):
    output = output.cpu()
    target = target.cpu()
    with torch.no_grad():
        return sklearn.metrics.roc_auc_score(target.numpy(), output.numpy())
