# ========================================================
#             Media and Cognition
#             Homework 3 Support Vector Machine
#             data_preprocess.py - Using pretrained convolutional layers to extract feature,
#                                   and using PCA for dimensionality reduction
#             Student ID: 2021012824
#             Name: 李沐晟
#             Tsinghua University
#             (C) Copyright 2024
# ========================================================

#用于数据预处理，从图像特征提取中改编而来，将特征向量改为嵌入向量。

import argparse
import os

import matplotlib.pyplot as plt
import torch
# import torchvision.transforms as transforms
# from PIL import Image
from zhipuai import ZhipuAI

# from networks import Classifier

client = ZhipuAI(api_key="80cba4165a53e1602e6d631cbd0caef9.YBb0NWpvyZq7WzIe")

def get_vector(input):
    response = client.embeddings.create(
        model="embedding-2",
        input=input,
    )
    return response.data[0].embedding

def preprocess(data_root, classes):
    # TODO 1: Using PCA to reduce the dimensionality of 2048 point features extracted by convolution

    # ===============  process training dataset ======================
    print("Start preprocessing the training dataset !!!")
    train_data, train_label = loaddata(data_root, 'train', classes)

    # calculate the mean and PCA projection matrix
    data_mean, u = PCA(train_data, 2)

    # TODO: using PCA to compress the dimensionality of the train_data after subtracting the mean vector
    centered_train_data = train_data - data_mean
    train_data_pca = 200*torch.mm(centered_train_data, u) #矩阵相乘，降为2维特征

    visualize(train_data_pca, train_label, "train",classes)
    savedata(train_data_pca, train_label, data_root+"/train"+str(classes[0])+str(classes[1])+".pt")
    print("training dataset saved !!!")

    # ===============  process validation dataset ======================
    print("Start preprocessing the validation dataset!!!")
    val_data, val_label = loaddata(data_root, 'val', classes)

    # TODO: using PCA to compress the dimensionality of the val_data after subtracting the mean vector
    centered_val_data = val_data - data_mean
    val_data_pca = 200*torch.mm(centered_val_data, u) #矩阵相乘，降为2维特征

    visualize(val_data_pca, val_label, "val",classes)
    savedata(val_data_pca, val_label, data_root+"/val"+str(classes[0])+str(classes[1])+".pt")
    print("validation dataset saved !!!")

    # ===============  process testing dataset ======================
    print("Start preprocessing the testing dataset!!!")
    test_data, test_label = loaddata(data_root, 'test', classes)

    # TODO: using PCA to compress the dimensionality of the test_data after subtracting the mean vector
    centered_test_data = test_data - data_mean
    test_data_pca = 200*torch.mm(centered_test_data, u) #矩阵相乘，降为2维特征

    visualize(test_data_pca, test_label, "test",classes)
    savedata(test_data_pca, test_label, data_root+"/test"+str(classes[0])+str(classes[1])+".pt")
    print("testing dataset saved !!!")


def savedata(data, label, save_path):
    save_dict = {
        'data': data,
        'label': label
    }
    torch.save(save_dict, save_path)


def visualize(datas, labels, mode,classes):
    """
    Display feature points after dimensionality reduction
    -------------------------------
    :param datas: the samples after dimensionality reduction, with the shape of [N, 2]
    :param labels: the labels (chosen from {-1, +1}) corresponding to the samples
    :param mode: chosen from {'train', 'val', 'test'}
    :return:
    """
    plt.figure()
    for idx in range(datas.shape[1]):
        plt.scatter(datas[labels == 2*idx-1, 0], datas[labels == 2*idx-1, 1], label=(2*idx-1))
    plt.legend()
    plt.title(mode+str(classes[0])+str(classes[1]))
    plt.show()


def PCA(data, dim=2):
    """
    calculate the mean value of the data and the projection matrix for PCA
    :param data: the sample features extracted by the pretrained network in homework2, with the shape of [N, 2048]
    :param dim: the data dimension after projection
    :return:
        data_mean: the mean value of the data
        u: the projection matrix for PCA, with the shape of [2048, dim]
    """
    # TODO 2: complete the algorithm of PCA, calculate the mean value of the data and the projection matrix

    # TODO: compute the mean of train_data
    data_mean = torch.mean(data, dim=0)
    # TODO: compute the covariance matrix of train_data
    centered_data = data - data_mean
    data_cov = torch.mm(centered_data.t(), centered_data) / (data.size(0) - 1)
    # TODO: compute the SVD decompositon of data_cov using torch.linalg.svd
    # reference: https://pytorch.org/docs/1.11/generated/torch.linalg.svd.html
    u, s, v = torch.linalg.svd(data_cov)
    u = u[:, :dim] #选取最大特征值对应的特征向量
    # TODO: return the proper 'data_mean' and 'u[]'
    return data_mean, u


def loaddata(data_root, mode, classes):
    """
    load one dataset, then do the embedding operataion to the texts
    :param data_root: the path of the dataset
    :param mode: chosen from {'train', 'val', 'test'}
    :param classes: two classes that need to be classified
    :return:
        datas: the samples of extracted features with the shape of [N, 2048]
        labels: the corresponding labels for each sample (chosen from {-1, +1}), with the shape of [N]
    """
    assert len(classes) == 2
    datas = []
    labels = []
    for idx in range(len(classes)):
        open_path=data_root + '/' + mode + '/' + classes[idx] + '.txt'
        with open(open_path,encoding='utf-8') as f:
            for text in f:
                data=torch.tensor(get_vector(text))
                label = torch.tensor(2 * idx - 1)
                datas.append(data)
                labels.append(label)
    return torch.stack(datas), torch.tensor(labels)


    # for idx in range(len(classes)):
    #     for img in os.listdir(data_root + '/' + mode + '/' + classes[idx]):
    #         data = readimg(pre_conv, data_root + '/' + mode + '/' + classes[idx] + '/' + img, image_size)
    #         label = 2 * idx - 1
    #         datas.append(data)
    #         labels.append(label)
    


# def readimg(pre_conv, filepath, image_size):
#     """
#     Read one image and use pretrained network to extract the feature
#     --------------------------
#     :param pre_conv: pretrained network in homework 2
#     :param filepath: the file path of one image
#     :param image_size: the preset size that each image try to zoom to
#     :return:
#         data: the extracted feature with the length of 2048
#     """
#     img_pil = Image.open(filepath).convert('RGB')
#     img_pil = img_pil.resize(image_size)
#     img_transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(0.5, 0.5),
#                                         ])
#     img_tensor = img_transform(img_pil)
#     data = pre_conv(img_tensor.unsqueeze(0)).reshape(-1)

#     return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_net", type=str, default="checkpoints/bn/ckpt_epoch_15.pth",
    #                     help="the filepath of the pretrained network in homework 2")
    parser.add_argument("--data_root", type=str, default="data", help="the path of all datasets")
    # parser.add_argument("--image_size", type=tuple, default=(32, 32),
    #                     help="the preset size that each image try to zoom to")
    parser.add_argument("--classes", default="12", help="two classes that need to be classified")

    args = parser.parse_args()

    # pretrained_checkpoint = torch.load(args.pretrained_net, map_location="cpu")
    # configs = pretrained_checkpoint["configs"]
    # cls = Classifier(
    #     configs["in_channels"],
    #     configs["num_classes"],
    #     configs["use_batch_norm"],
    #     configs["use_stn"],
    #     configs["dropout_prob"],
    # )
    # cls.load_state_dict(pretrained_checkpoint["model_state"])
    # for param in cls.parameters():
    #     param.requires_grad = False
    # conv = cls.conv_net

    preprocess(args.data_root, args.classes)
