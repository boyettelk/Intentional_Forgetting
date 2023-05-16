from rtpt import RTPT
import numpy as np
import os, random, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.datasets as datasets
from model_nesy import *
from data import traindataset0, traindataset1, traindataset2
from data import valdataset0, valdataset1, valdataset2
from model import CNNModel, ResNet, BasicBlock
from utils import *

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=20,
    help='number of epochs to train our network for')
parser.add_argument('-m', '--model', type=str, default='Resnet', 
    help='select the model you would like to run training with')
args = vars(parser.parse_args())


def set_seed(seed=42):
    """
    Set random seeds for all possible random processes.
    :param seed: int
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopper:
    def __init__(self, patience=3, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.new_count = -5
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, valid_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # also stop training if validation has consistently reached very low values
            self.new_count += 1
            if self.new_count >= self.patience:
                return True
        elif min(valid_loss) < validation_loss:
            if (validation_loss - min(valid_loss)) > 0.1:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False

# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(image)

        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # backpropagation
        loss.backward()

        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        epoch_y_pred, epoch_y_true = [], []
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(image)

            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)

            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    return epoch_loss, epoch_acc

def run(train_loaders, val_loaders):

    M_l = []
    M_a = []

    for i_train, x in enumerate(train_loaders):
        for i_val, y in enumerate(val_loaders):
            train_data = x
            val_data = y
            #matrix_id = [i_train, i_val] #sanity check

            #### choose model type
            if args['model'] == 'Resnet':
                model = ResNet(img_channels=3, num_layers=18, num_classes=2, block=BasicBlock).to(device)
                model_name = 'Resnet18'
            elif args['model'] == 'CNN':
                model = CNNModel().to(device)
                model_name = 'CNN'
            elif args['model'] == 'Slot':
                model = NeSyConceptLearner(n_classes=3, n_slots=10, n_iters=3, n_attr=18, n_set_heads=4, set_transf_hidden=128,
                                        category_ids = [3, 6, 8, 10, 17], device='cuda')
                model_name = 'Slot_Attention'

            print(f"Computation device: {device}\n")
            print(model_name, model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"{total_params:,} total parameters.")
            total_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{total_trainable_params:,} training parameters.")
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            ### DO EXPERIMENT

            # lists to keep track of losses and accuracies
            train_loss, valid_loss = [], []
            train_acc, valid_acc = [], []

            # start the training
            early_stopper = EarlyStopper(patience=5, min_delta=1)

            for epoch in range(epochs):
                print(f"[INFO]: Epoch {epoch+1} of {epochs}")

                train_epoch_loss, train_epoch_acc = train(model, train_data, 
                                                        optimizer, criterion)
                valid_epoch_loss, valid_epoch_acc = validate(model, val_data,  
                                                            criterion)
                
                train_loss.append(train_epoch_loss)
                valid_loss.append(valid_epoch_loss)
                train_acc.append(train_epoch_acc)
                valid_acc.append(valid_epoch_acc)

                if early_stopper.early_stop(valid_epoch_loss, valid_loss):
                    print("We are at epoch:", epoch)
                    break

                else:
                    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
                    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
                    print('-'*50)

                    #rtpt update
                    rtpt.step()

            #end epochs
            acc = np.average(valid_acc)
            loss = np.average(valid_loss)
            l = loss
            a = acc
            # initial_acc = valid_acc[0]
            # final_acc = valid_acc[-1]

            # initial_loss = valid_loss[0]
            # final_loss = valid_loss[-1]

            # l = final_loss 
            # a = final_acc

            time.sleep(2)
            M_l.append(l)
            M_a.append(a)

    loss_arr = np.asarray(M_l)
    acc_arr = np.asarray(M_a)
    new_arr_l = loss_arr.reshape(3,3)
    new_arr_a = acc_arr.reshape(3,3)
    output_loss = new_arr_l.T
    output_acc = new_arr_a.T
    return output_loss, output_acc
        


if __name__ == "__main__":
    #training params
    BATCH_SIZE = args['batchsize']
    lr = 1e-3
    epochs = args['epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers':1, 'pin_memory':True} if device=='cuda' else {}

    #RTPT
    rtpt = RTPT(name_initials='LB', experiment_name='IF_Training', max_iterations=epochs*9)
    rtpt.start()

    set_seed(seed=42)

    #get dataloaders
    train0_loader = DataLoader(dataset= traindataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train1_loader = DataLoader(dataset= traindataset1, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train2_loader = DataLoader(dataset= traindataset2, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val0_loader = DataLoader(dataset= valdataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val1_loader = DataLoader(dataset= valdataset1, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val2_loader = DataLoader(dataset= valdataset2, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    train_loaders = [train0_loader, train1_loader, train2_loader]
    val_loaders = [val0_loader, val1_loader, val2_loader]

    loss, acc = run(train_loaders, val_loaders)
    #plot_heatmaps(acc, loss, model_name)
    create_heatmap_acc(acc, model_name)
    create_heatmap_loss(loss, model_name)