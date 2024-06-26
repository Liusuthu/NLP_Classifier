import torch
from torch.autograd import gradcheck

from svm_hw import SVM_HINGE, Hinge, LinearFunction


def run(batch_size=50):
    model = SVM_HINGE(2, C=1.0).double()#这里取input_size=2是因为后续特征降维成2维了
    x = torch.randn(batch_size, 2, requires_grad=False).double()
    W = torch.randn(1, 2, requires_grad=True).double()
    b = torch.zeros(1, requires_grad=True).double()
    test = gradcheck(LinearFunction.apply, (x, W, b), eps=1e-6, atol=1e-4)
    if test:
        print('Linear successully tested!')
    output = torch.randn(batch_size, 1, requires_grad=True).double()
    W = torch.randn(1, 2, requires_grad=True).double()
    labels = torch.ones(batch_size, requires_grad=False).double()
    C = torch.tensor([[1.0]], requires_grad=False).double()
    test = gradcheck(Hinge.apply, (output, W, labels, C), eps=1e-6, atol=1e-5)
    if test:
        print('Hinge successfully tested！')
    x = torch.randn(batch_size, 2, requires_grad=False).double()
    labels = torch.ones(batch_size, requires_grad=False).double()
    try:
        output, loss = model(x, labels)
        assert model.W.requires_grad is True
        assert model.b.requires_grad is True
        print('SVM_HINGE successfully tested！')
    except:
        raise Exception('Failed testing SVM_HINGE!')


if __name__ == '__main__':
    run()