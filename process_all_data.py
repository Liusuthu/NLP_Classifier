import argparse
import os

import matplotlib.pyplot as plt
import torch
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="80cba4165a53e1602e6d631cbd0caef9.YBb0NWpvyZq7WzIe")

def get_vector(input):
    response = client.embeddings.create(
        model="embedding-2",
        input=input,
    )
    return response.data[0].embedding


def preprocess(data_root, classes):
    # ===============  process training dataset ======================
    print("Start preprocessing the training dataset !!!")
    train_data, train_label = loaddata(data_root, 'train', classes)

    # calculate the mean and PCA projection matrix
    data_mean, u = PCA(train_data, 2)

    # using PCA to compress the dimensionality of the train_data after subtracting the mean vector
    centered_train_data = train_data - data_mean
    train_data_pca = 200*torch.mm(centered_train_data, u) #矩阵相乘，降为2维特征

    visualize(train_data_pca, train_label, "train")
    savedata(train_data_pca, train_label, data_root+"/train"+str(classes[0])+str(classes[1])+".pt")
    print("training dataset saved !!!")

    # ===============  process validation dataset ======================
    print("Start preprocessing the validation dataset!!!")
    val_data, val_label = loaddata(data_root, 'val')

    # using PCA to compress the dimensionality of the val_data after subtracting the mean vector
    centered_val_data = val_data - data_mean
    val_data_pca = 200*torch.mm(centered_val_data, u) #矩阵相乘，降为2维特征

    visualize(val_data_pca, val_label, "val")
    savedata(val_data_pca, val_label, data_root+"/val"+str(classes[0])+str(classes[1])+".pt")
    print("validation dataset saved !!!")

    # ===============  process testing dataset ======================
    print("Start preprocessing the testing dataset!!!")
    test_data, test_label = loaddata(data_root, 'test')

    #using PCA to compress the dimensionality of the test_data after subtracting the mean vector
    centered_test_data = test_data - data_mean
    test_data_pca = 200*torch.mm(centered_test_data, u) #矩阵相乘，降为2维特征

    visualize(test_data_pca, test_label, "test")
    savedata(test_data_pca, test_label, data_root+"/test"+str(classes[0])+str(classes[1])+".pt")
    print("testing dataset saved !!!")


def savedata(data, label, save_path):
    save_dict = {
        'data': data,
        'label': label
    }
    torch.save(save_dict, save_path)


def visualize(datas, labels, mode):
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
    plt.title(mode)
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
    # compute the mean of train_data
    data_mean = torch.mean(data, dim=0)
    # compute the covariance matrix of train_data
    centered_data = data - data_mean
    data_cov = torch.mm(centered_data.t(), centered_data) / (data.size(0) - 1)
    # compute the SVD decompositon of data_cov using torch.linalg.svd
    # reference: https://pytorch.org/docs/1.11/generated/torch.linalg.svd.html
    u, s, v = torch.linalg.svd(data_cov)
    u = u[:, :dim] #选取最大特征值对应的特征向量
    # return the proper 'data_mean' and 'u[]'
    return data_mean, u


def loaddata(data_root, mode,):
    """
    load one dataset, then do the embedding operataion to the texts
    :param data_root: the path of the dataset
    :param mode: chosen from {'train', 'val', 'test'}
    :param classes: two classes that need to be classified
    :return:
        datas: the samples of extracted features with the shape of [N, 2048]
        labels: the corresponding labels for each sample (chosen from {-1, +1}), with the shape of [N]
    """
    datas=[]
    labels=[]
    for i in range(4):
        open_path=data_root + '/' + mode + '/' + str(i+1) + '.txt'
        with open(open_path,encoding='utf-8') as f:
            for text in f:
                data=torch.tensor(get_vector(text))
                label=torch.tensor(i+1)
                datas.append(data)
                labels.append(label)
    return torch.stack(datas), torch.tensor(labels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="the path of all datasets")
    parser.add_argument("--classes", default="12", help="two classes that need to be classified")

    args = parser.parse_args()


    preprocess(args.data_root, args.classes)
