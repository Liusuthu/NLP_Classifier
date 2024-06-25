# ========================================================
#             Media and Cognition
#             Homework 3 Support Vector Machine
#             train_svm.py - Train svm model for traffic sign
#             Student ID: 2021012824
#             Name: 李沐晟
#             Tsinghua University
#             (C) Copyright 2024
# ========================================================

# ==== Part 1: import libs
import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import Traffic_Dataset
from svm_hw import SVM_HINGE


# ==== Part 2: training and validation
def train(
    data_root,
    feature_channel,
    batch_size,
    n_epoch,
    lr,
    C,
    #model_save_path,
    device,
    classes,
):
    """
    The main training procedure of SVM model
    ----------------------------
    :param data_root: path to the root directory of dataset
    :param feature_channel: number of feature channels for SVM input
    :param batch_size: batch size of training
    :param n_epoch: number of training epochs
    :param lr: learning rate
    :param C: regularization coefficient in hinge loss
    :param model_save_path: path to save SVM model
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    """


    model_save_path="checkpoints/svm"+str(classes[0])+str(classes[1])+".pth"
    # TODO 1: construct training and validation data loader with 'Traffic_Dataset' and DataLoader, and set proper values for 'batch_size' and 'shuffle'
    train_data = Traffic_Dataset(data_root+'/train'+str(classes[0])+str(classes[1])+'.pt')
    train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=2,shuffle=True)
    val_data = Traffic_Dataset(data_root+'/val'+str(classes[0])+str(classes[1])+'.pt')
    val_loader = DataLoader(val_data,batch_size=batch_size,num_workers=2,shuffle=True)

    # scale the regularization coefficient
    C = C * len(train_loader)

    # TODO: initialize the SVM model
    svm = SVM_HINGE(in_channels=feature_channel,C=C)#param in_channels: number of feature channels for SVM input

    # TODO: put the model on CPU or GPU
    svm = svm.to(device) 

    # TODO: define the Adam optimizer
    optimizer = torch.optim.Adam(svm.parameters(), lr=lr)

    # to save the training loss, training accuracy, validation accuracy, and the epoch index of each training epoch
    train_loss = []
    train_acc = []
    val_acc = []
    epochs = []

    for epoch in range(n_epoch):
        # TODO: save the index of current epoch in the array 'epochs'
        epochs.append(epoch+1)

        # TODO 2: ========================= training =======================
        # TODO: set the model in training mode
        svm.train()

        # to calculate and save the training loss and training accuracy
        total_loss = 0.  # to save total training loss in one epoch
        n_correct = 0.  # number of images that are correctly classified
        n_feas = 0.  # number of total images

        # TODO: get a batch of data; you may need enumerate() to iteratively get data from 'train_loader'.
        # you can refer to previous homework, for example hw2
        for step, (input, label) in enumerate(train_loader):
            #print("step:",step,"---------------------")
            # TODO: set data type (.float()) and device (.to())
            input, label = (
                input.float().to(device),
                label.float().to(device),
            )
            #print("input:",input,",label:",label)
            # print("input:",input)

            # TODO: clear gradients in the optimizer
            optimizer.zero_grad()

            # TODO: run the model with hinge loss; the model needs two inputs: feas and labels
            out, loss = svm(input,label)#有问题——输出全是out
            #print("svm_out:",out)
            # print("out:",out)
            # print("label:",label)

            # TODO: back-propagation on the computation graph
            loss.backward()

            # TODO: sum up of total loss, loss.item() return the value of the tensor as a standard python number
            total_loss += loss.item()

            # TODO: call a function to update the parameters of the models 更新模型参数
            optimizer.step()

            # TODO: sum up the number of images correctly recognized. note the shapes of 'out' and 'labels' are different
            n_correct += torch.sum((out.view(1,-1)==label).float()).item()#这里存在维度问题，故使用view()进行对齐
            #print("n_correct:",n_correct)
            # TODO: sum up the total image number
            n_feas += input.size(0)
            #print("n_feas:",n_feas)

        # average of the total loss for iterations
        #print("n_correct:",n_correct,"n_feas:",n_feas)
        acc = torch.tensor(100 * n_correct / n_feas)#这里做了一些更改
        avg_loss = total_loss / len(train_loader)
        train_acc.append(acc.cpu().numpy())
        train_loss.append(avg_loss)
        print('Epoch {:02d}: loss = {:.3f}, training accuracy = {:.1f}%'.format(epoch + 1, avg_loss, acc))

        # TODO 3: ========================== Validation ======================================

        # TODO: set the model in evaluation mode
        svm.eval()

        # to calculate and save the validation accuracy
        n_correct = 0.  # number of images that are correctly classified
        n_feas = 0.  # number of total images

        with torch.no_grad():  # we do not need to compute gradients during validation
            # TODO: inference on the validation dataset, similar to the training stage but use 'val_loader'.
            for step, (input, label) in enumerate(val_loader):
                # TODO: set data type (.float()) and device (.to())
                input, label = (
                    input.float().to(device),
                    label.float().to(device),
                )

                # TODO: run the model; at the validation step, the model only needs one input: feas
                # _ refers to a placeholder, which means we do not need the second returned value during validating
                out, _ = svm(input,label)

                # TODO: sum up the number of images correctly recognized. note the shapes of 'out' and 'labels' are different
                n_correct += torch.sum((out.view(1,-1) == label).float()).item()

                # TODO: sum up the total image number
                n_feas += input.size(0)

        # show prediction accuracy
        #print("n_correct:",n_correct,"n_feas:",n_feas)
        acc = torch.tensor(100 * n_correct / n_feas)#这里做了一些更改
        print('Epoch {:02d}: validation accuracy = {:.1f}%'.format(epoch + 1, acc))
        val_acc.append(acc.cpu().numpy())

    # save model parameters in a file
    torch.save({'state_dict': svm.state_dict(),
                'configs': {
                    'feature_channel': feature_channel,
                    'C': C}
                }, model_save_path)
    print('Model saved in {}\n'.format(model_save_path))

    W = svm.W.data.cpu()
    b = svm.b.data.cpu()

    # TODO 4: calculate the index of support vectors in training samples using 'train_data.datas' and 'train_data.labels'
    # 'sv' should be a list in python structure with the shape of [K], where K is the number of support vectors.
    sv = []
    for i, (data, label) in enumerate(zip(train_data.datas, train_data.labels)): # 使用zip同时遍历data和labels
        data, label = data.to(device), label.to(device)
        output = svm(data.unsqueeze(0))[0].item()
        if abs(output) < 1.0:  # non-zero margin则是支持向量
            sv.append(i)





    plot(train_loss, train_acc, val_acc, epochs,classes)
    plot_feature(train_features=train_data.datas, val_features=val_data.datas, train_labels=train_data.labels,
                 val_labels=val_data.labels, sv=sv, W=W, b=b,classes=classes)


def plot_feature(train_features, val_features, train_labels, val_labels, sv, W, b,classes):
    """
    Draw the samples,SVM decision boundary, and support vectors
    ---------------------
    :param train_features: training samples with the shape of [B, 2]
    :param val_features: validation samples with the shape of [B, 2]
    :param train_labels: the labels (chosen from{-1, +1}) corresponding to training samples, with the shape of [B, 1]
    :param val_labels: the labels (chosen from{-1, +1}) corresponding to validation samples, with the shape of [B, 1]
    :param sv: a list with the index of support vectors in training samples, with the shape of [K] (K is the number of support vectors)
    :param W: the weight vector of SVM decision boundary (W^Tx + b), with the shape of [1, feature_channel]
    :param b: the bias of SVM decision boundary (W^Tx + b), with the shape of [1,]
    """
    train_labels = (train_labels > 0.0).int()
    val_labels = (val_labels > 0.0).int()
    train_labels[sv] = 2
    foreground = list(set([i for i in range(train_labels.shape[0] // 2)]) - set(sv))
    foreground_sv = list(set([i for i in range(train_labels.shape[0] // 2)]) - set(foreground))
    background = list(set([i + train_labels.shape[0] // 2 for i in range(train_labels.shape[0] // 2)]) - set(sv))
    background_sv = list(set([i + train_labels.shape[0] // 2 for i in range(train_labels.shape[0] // 2)]) - set(background))
    f, ax = plt.subplots()
    plt.title("training dataset"+str(classes[0])+str(classes[1]))
    ax.scatter(train_features[foreground, 0], train_features[foreground, 1], marker='.', c='r', label="-1")
    ax.scatter(train_features[foreground_sv, 0], train_features[foreground_sv, 1], marker='.', c='darkorange',
               label="-1 (support vector)")
    ax.scatter(train_features[background, 0], train_features[background, 1], marker='x', c='b', label="+1")
    ax.scatter(train_features[background_sv, 0], train_features[background_sv, 1], marker='x', c='c',
               label="+1 (support vector)")
    x = np.linspace(-20, 20, 100)
    ax.plot(x, -W[0, 0] / W[0, 1] * x - b / W[0, 1], c='y')
    ax.legend(loc="best")
    plt.ylim([-30, 30])
    plt.show()
    f, ax = plt.subplots()
    plt.title("validation dataset"+str(classes[0])+str(classes[1]))
    foreground_val = [i for i in range(val_labels.shape[0] // 2)]
    background_val = [i + val_labels.shape[0] // 2 for i in range(val_labels.shape[0] // 2)]
    ax.scatter(val_features[foreground_val, 0], val_features[foreground_val, 1], marker='.', c='r', label="-1")
    ax.scatter(val_features[background_val, 0], val_features[background_val, 1], marker='x', c='b', label="+1")
    x = np.linspace(-20, 20, 100)
    ax.plot(x, -W[0, 0] / W[0, 1] * x - b / W[0, 1], c='y')
    ax.legend(loc="best")
    plt.ylim([-30, 30])
    plt.show()


def plot(train_loss, train_acc, val_acc, epochs,classes):
    """
    Draw loss and accuracy curve
    ------------------
    :param train_loss: a list with loss of each training epoch
    :param train_acc: a list with accuracy on training dataset of each training epoch
    :param val_acc: a list with accuracy on validation dataset of each training epoch
    :param epochs: a list with the index of all training epochs
    """

    # draw the training loss curve
    f, ax = plt.subplots()
    plt.title("Training Loss"+str(classes[0])+str(classes[1]))
    ax.plot(epochs, train_loss, color="tab:blue")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Loss")
    ax.legend(["training loss"], loc="best")
    plt.show()

    # draw the accuracy curve
    f, ax = plt.subplots()
    plt.title("Training and Validation Accuracy"+str(classes[0])+str(classes[1]))
    ax.plot(epochs, train_acc, color="tab:orange")
    ax.plot(epochs, val_acc, color="tab:green")
    ax.legend(["training accuracy","validation accuracy"], loc="best")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 101)
    plt.show()


if __name__ == "__main__":
    # set random seed for reproducibility
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # set configurations of the model and training process
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="file list of training image paths and labels",)
    parser.add_argument("--n_epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=20, help="training batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--C", type=float, default=1e-3, help="regularization coefficient in hinge loss")
    parser.add_argument("--device", type=str, help="cpu or cuda")
    parser.add_argument("--feature_channel", type=int, default=2, help="number of pre-extracted feature channel by pretrained network")
    #parser.add_argument("--model_save_path", type=str, default="checkpoints/svm.pth", help="path to save SVM model")
    parser.add_argument("--classes", default="12", help="two classes that need to be classified")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # run the training procedure
    train(
        data_root=args.data_root,
        feature_channel=args.feature_channel,
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
        lr=args.lr,
        C=args.C,
        #model_save_path=args.model_save_path,
        device=args.device,
        classes=args.classes,
    )










