# ==== Part 1: import libs
import argparse

import torch
from torch.utils.data import DataLoader

from datasets import Traffic_Dataset
from svm_hw import SVM_HINGE


# ==== Part 2: testing
def test(
    data_root,
    #model_save_path,
    device,
    classes,
):
    """
    The main testing procedure of SVM model
    ----------------------------
    :param data_root: path to the root directory of dataset
    :param model_save_path: path to pretrained SVM model
    :param device: device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    """
    
    model_save_path="checkpoints/svm"+str(classes[0])+str(classes[1])+".pth"
    #  =================== load the pretrained SVM model ==================================

    test_data = Traffic_Dataset(data_root+'/test'+str(classes[0])+str(classes[1])+'.pt')
    test_loader = DataLoader(test_data,batch_size=1,num_workers=2,shuffle=False)

    #  load state dictionary of pretrained SVM model
    model_svm = torch.load(model_save_path)

    # initialize the SVM model using 'model_svm["configs"]["feature_channel"]' and 'model_svm["configs"]["C"]'
    svm = SVM_HINGE(in_channels=model_svm['configs']['feature_channel'], C=model_svm['configs']['C'])

    #load model parameters (model_svm['state_dict']) we saved in model_path using svm.load_state_dict()
    svm.load_state_dict(model_svm['state_dict'])

    # put the model on CPU or GPU
    svm = svm.to(device)

    #  ================================ testing ==============================================

    # set the model in evaluation mode
    svm.eval()

    # to calculate and save the testing accuracy
    n_correct = 0.  # number of images that are correctly classified
    n_feas = 0.  # number of total images

    with torch.no_grad():  # we do not need to compute gradients during validation
        # TODO: inference on the testing dataset, similar to the training stage but use 'test_loader'.
        for input, label in test_loader:
            # TODO: set data type (.float()) and device (.to())
            input, label = (
                input.float().to(device),
                label.float().to(device),
            )

            # run the model; at the validation step, the model only needs one input: feas
            # _ refers to a placeholder, which means we do not need the second returned value during validating
            out, _ = svm(input,label)

            # sum up the number of images correctly recognized. note the shapes of 'out' and 'labels' are different
            n_correct += torch.sum((out.view(1,-1)==label).float()).item()

            # sum up the total image number
            n_feas += input.size(0)

    # show prediction accuracy
    acc = 100 * n_correct / n_feas
    print('Test accuracy = {:.1f}%'.format(acc))


if __name__ == "__main__":
    # set configurations of the testing process
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="file list of training image paths and labels")
    parser.add_argument("--device", type=str, help="cpu or cuda")
    parser.add_argument("--classes", default="12", help="two classes that need to be classified")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # run the testing procedure
    test(
        data_root=args.data_root,
        device=args.device,
        classes=args.classes,
    )









