import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# from PIL import Image
from zhipuai import ZhipuAI

from datasets import Traffic_Dataset
from svm_hw import SVM_HINGE

#用OvO投票法给出判决

client = ZhipuAI(api_key="80cba4165a53e1602e6d631cbd0caef9.YBb0NWpvyZq7WzIe")
test_list=['12','13','14','23','24','34']
model_save_path=["checkpoints/svm12.pth","checkpoints/svm13.pth","checkpoints/svm14.pth",
                    "checkpoints/svm23.pth","checkpoints/svm24.pth","checkpoints/svm34.pth"]
# test_data_path=[data_root+'/test12.pt',data_root+'/test13.pt',data_root+'/test14.pt',
#                 data_root+'/test23.pt',data_root+'/test24.pt',data_root+'/test34.pt']

def get_vector(input):
    response = client.embeddings.create(
        model="embedding-2",
        input=input,
    )
    return response.data[0].embedding

def predict_vote(data,device,mean,u):#传入的是1024维的data
    #对每个数据集(1~4)中的测试数据进行测试
    vote=[0,0,0,0]
    for i in range(6):
        #对每个样本，使用6个SVM进行测试后投票
        model_svm = torch.load(model_save_path[i])
        svm = SVM_HINGE(in_channels=model_svm['configs']['feature_channel'], C=model_svm['configs']['C'])
        svm.load_state_dict(model_svm['state_dict'])
        svm = svm.to(device)
        svm.eval()

        u_i=u[i]
        mean_i=mean[i]
        test_item=test_list[i]
        print("test_item:",test_item)
        with torch.no_grad():
            input=200*(data-mean_i)@(u_i)
            input.float().to(device)
            out, _ = svm(input,label=None)
            if out==-1:
                vote[int(test_item[0])-1]+=1
            else:
                vote[int(test_item[1])-1]+=1
    return vote.index(max(vote))+1

def test(
    data_root,
    device,
):
    print("开始测试...")

    #首先得到所有投影向量
    print("数据预处理...")
    train_data, train_label = loaddata(data_root, 'train', "12")
    data_mean12, u12 = PCA(train_data, 2)
    # print(data_mean12,data_mean12.shape)
    # print(u12,u12.shape)
    print("12预处理完毕")
    train_data, train_label = loaddata(data_root, 'train', "13")
    data_mean13, u13 = PCA(train_data, 2)
    print("13预处理完毕")
    train_data, train_label = loaddata(data_root, 'train', "14")
    data_mean14, u14 = PCA(train_data, 2)
    print("14预处理完毕")
    train_data, train_label = loaddata(data_root, 'train', "23")
    data_mean23, u23 = PCA(train_data, 2)
    print("23预处理完毕")
    train_data, train_label = loaddata(data_root, 'train', "24")
    data_mean24, u24 = PCA(train_data, 2)
    print("24预处理完毕")
    train_data, train_label = loaddata(data_root, 'train', "34")
    data_mean34, u34 = PCA(train_data, 2)
    print("34预处理完毕")
    mean=[data_mean12,data_mean13,data_mean14,data_mean23,data_mean24,data_mean34]
    u=[u12,u13,u14,u23,u24,u34]

    #进入测试环节
    total=0
    correct=0
    print("进入测试环节")
    for dataset_num in range(4):
        open_path=data_root + '/' + 'test' + '/' + str(dataset_num+1) + '.txt'
        with open(open_path,encoding='utf-8') as f:
            for text in f:
                total=total+1
                data=torch.tensor(get_vector(text))
                label = torch.tensor(dataset_num+1)
                pred_label=torch.tensor(predict_vote(data,device,mean,u))
                print("对于第",total,"个样本,真实label为",label,",判决结果为",pred_label)
                if pred_label==label:
                    correct=correct+1

    #测试结束并输出结果
    print("测试完毕...")
    acc = 100 * correct / total
    print('Test accuracy = {:.1f}%'.format(acc))







def loaddata(data_root, mode,classes):
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


if __name__ == "__main__":
    # set configurations of the testing process
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="file list of training image paths and labels")
    parser.add_argument("--device", type=str, default='cpu', help="cpu or cuda")
    #parser.add_argument("--model_save_path", type=str, default="checkpoints/svm.pth", help="path to save SVM model")
    #parser.add_argument("--classes", default="12", help="two classes that need to be classified")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # run the testing procedure
    test(
        data_root=args.data_root,
        #model_save_path=args.model_save_path,
        device=args.device,
        #classes=args.classes,
    )
