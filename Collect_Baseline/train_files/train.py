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
parser.add_argument('-e', '--epochs', type=int, default=30,
    help='number of epochs to train our network for')
parser.add_argument('-b', '--batchsize', type=int, default=64,
    help='number of images per batch')
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


def get_loaders(concat_type):
'''
Passes the training and validation dataloaders to the run fxn.
'''
    if concat_type == 3:
        train_data = ConcatDataset([traindataset0, traindataset1, traindataset2])
        all_train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        val_data = ConcatDataset([valdataset0, valdataset1, valdataset2])
        all_valid_loader = DataLoader(dataset= val_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

        train_lst = [all_train_loader]
        val_lst = [all_valid_loader]

    elif concat_type == 2:
        #train data
        train0_loader = DataLoader(dataset= traindataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        secondtr = ConcatDataset([traindataset0, traindataset1])
        train1_loader = DataLoader(dataset= secondtr, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        thirdtr = ConcatDataset([traindataset0, traindataset1, traindataset2])
        train2_loader = DataLoader(dataset= thirdtr, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        #next validity data
        valid0_loader = DataLoader(dataset= valdataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        secondv = ConcatDataset([valdataset0, valdataset1])
        valid1_loader = DataLoader(dataset= secondv, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        thirdv = ConcatDataset([valdataset0, valdataset1, valdataset2])
        valid2_loader = DataLoader(dataset= thirdv, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

        train_lst = [train0_loader, train1_loader, train2_loader]
        val_lst = [valid0_loader, valid1_loader, valid2_loader]

    elif concat_type == 1:
        train0_loader = DataLoader(dataset= traindataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        train1_loader = DataLoader(dataset= traindataset1, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        train2_loader = DataLoader(dataset= traindataset2, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

        valid0_loader = DataLoader(dataset= valdataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        secondv = ConcatDataset([valdataset0, valdataset1])
        valid1_loader = DataLoader(dataset= secondv, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        thirdv = ConcatDataset([valdataset0, valdataset1, valdataset2])
        valid2_loader = DataLoader(dataset= thirdv, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

        train_lst = [train0_loader, train1_loader, train2_loader]
        val_lst = [valid0_loader, valid1_loader, valid2_loader]
    
    return train_lst, val_lst

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

def run(concat_type):

    #### choose model type
    if args['model'] == 'Resnet':
        model = ResNet(img_channels=3, num_layers=18, num_classes=2, block=BasicBlock).to(device)
        model_name = 'Resnet18'
    elif args['model'] == 'CNN':
        model = CNNModel().to(device)
        model_name = 'CNN'
    elif args['model'] == 'Slot':
        model = NeSyConceptLearner(n_classes=2, n_slots=10, n_iters=3, n_attr=18, n_set_heads=4, set_transf_hidden=128,
                                category_ids = args.category_ids, device='cuda')
        model_name = 'Slot_Attention'
    

    print(f"Computation device: {device}\n")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    #get dataloaders
    train_lst, val_lst = get_loaders(concat_type)

    ### DO EXPERIMENT
    TL, VL, TA, VA = [],[],[],[]

    final_val_acc, initial_val_acc = [], []

    #set up for different time steps:
    for i in range(len(train_lst)):

        train_loader = train_lst[i]
        valid_loader = val_lst[i]

        # lists to keep track of losses and accuracies
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        # start the training
        early_stopper = EarlyStopper(patience=5, min_delta=1)

        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")

            train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                    optimizer, criterion)
            valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
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
                time.sleep(5)

                #rtpt update
                rtpt.step()

        TL.append(train_loss)
        VL.append(valid_loss)
        TA.append(train_acc)
        VA.append(valid_acc)

        #end of epoch loops        
        print("End of training timestep")

    #end of train_timesteps
    print("We have run through all training timesteps.")
    if not os.path.exists('./output'):
        os.makedirs('./output')

    # save the trained model weights
    print('Saving the model')
    save_model(model_name, epochs, model, optimizer, criterion)

    # save the loss and accuracy plots
    print('Plotting Accuracy and Loss')
    save_plots(TL, VL, TA, VA, model_name, concat_type)
    print('TRAINING COMPLETE')

if __name__ == "__main__":
    #training params
    BATCH_SIZE = args['batchsize']
    lr = 1e-3
    epochs = args['epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers':1, 'pin_memory':True} if device=='cuda' else {}
    
    #RTPT
    rtpt = RTPT(name_initials='LB', experiment_name='ResNet_Training', max_iterations=epochs*3*3)
    rtpt.start()

    set_seed(seed=42)

    #train for each concat type
    for ct in [1, 2, 3]:
        run(ct)

