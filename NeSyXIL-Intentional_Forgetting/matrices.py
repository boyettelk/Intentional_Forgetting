import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import random
import argparse
import glob
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn import metrics
from tqdm import tqdm

import NeSyConceptLearner.src.model as model
import NeSyConceptLearner.src.utils as utils
import IF_data as data
from xil_losses import LexiLoss
from rtpt import RTPT

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)


class EarlyStopper:
    def __init__(self, patience=3):
        self.patience = patience
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

# -----------------------------------------
# - Define basic and data related methods -
# -----------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--mode", type=str, required=True, help="train, test, or plot")
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--l2_grads", type=float, default=1, help="Right for right reason weight"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=["clevr-hans-state"],
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")

    parser.add_argument("--data-dir", type=str, help="Directory to data")
    parser.add_argument("--fp-ckpt", type=str, default=None, help="checkpoint filepath")

    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')

    args = parser.parse_args()

    # hard set !!!!!!!!!!!!!!!!!!!!!!!!!
    args.n_heads = 4
    args.set_transf_hidden = 128

    assert args.data_dir.endswith(os.path.sep)
    args.conf_version = args.data_dir.split(os.path.sep)[-2]
    args.name = args.name + f"-{args.conf_version}"

    if args.mode == 'test' or args.mode == 'plot':
        assert args.fp_ckpt

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    utils.seed_everything(args.seed)

    return args


def get_confusion_from_ckpt(net, test_loader, criterion, args, datasplit, writer=None):

    true, pred, true_wrong, pred_wrong = run_test_final(net, test_loader, criterion, writer, args, datasplit)
    precision, recall, accuracy, f1_score = utils.performance_matrix(true, pred)

    # Generate Confusion Matrix
    if writer is not None:
        utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, f'Confusion_matrix_normalize_{datasplit}.pdf'))
        utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, f'Confusion_matrix_{datasplit}.pdf'))
    else:
        utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    f'Confusion_matrix_normalize_{datasplit}.pdf'))
        utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    f'Confusion_matrix_{datasplit}.pdf'))
    return accuracy

# -----------------------------------------
# - Define Train/Test/Validation methods -
# -----------------------------------------
def run_test_final(net, loader, criterion, writer, args, datasplit):
    net.eval()

    running_corrects = 0
    running_loss=0
    pred_wrong = []
    true_wrong = []
    preds_all = []
    labels_all = []
    with torch.no_grad():

        for i, sample in enumerate(tqdm(loader)):
            # input is either a set or an image
            imgs, target_set, img_class_ids, img_ids, _, table_expl = map(lambda x: x.cuda(), sample)
            img_class_ids = img_class_ids.long()

            # forward evaluation through the network
            output_cls, output_attr = net(imgs)
            # for training only set transformer given GT symbols
            # output_cls = net.set_cls(target_set)

            # class prediction
            _, preds = torch.max(output_cls, 1)

            labels_all.extend(img_class_ids.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

            running_corrects = running_corrects + torch.sum(preds == img_class_ids)
            loss = criterion(output_cls, img_class_ids)
            running_loss += loss.item()
            preds = preds.cpu().numpy()
            target = img_class_ids.cpu().numpy()
            preds = np.reshape(preds, (len(preds), 1))
            target = np.reshape(target, (len(preds), 1))

            for i in range(len(preds)):
                if (preds[i] != target[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(target[i])

        bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

        if writer is not None:
            writer.add_scalar(f"Loss/{datasplit}_loss", running_loss / len(loader), 0)
            writer.add_scalar(f"Acc/{datasplit}_bal_acc", bal_acc, 0)

        return labels_all, preds_all, true_wrong, pred_wrong


def run(net, loader, optimizer, criterion, split, writer, args, train=False, plot=False, epoch=0):
    if train:
        net.img2state_net.eval()
        net.set_cls.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc="{1} E{0:02d}".format(epoch, "train" if train else "val "),
    )
    running_loss = 0
    preds_all = []
    labels_all = []
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
        # input is either a set or an image
        imgs, target_set, img_class_ids, img_ids, _, table_expl = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr = net(imgs)
        # for training only set transformer given GT symbols
        # output_cls = net.set_cls(target_set)

        # class prediction
        _, preds = torch.max(output_cls, 1)

        loss = criterion(output_cls, img_class_ids)

        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        labels_all.extend(img_class_ids.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        # Plot predictions in Tensorboard
        if plot and not(i % iters_per_epoch):
            utils.write_expls(net, loader, f"Expl/{split}", epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )

    return bal_acc, running_loss / len(loader)


def run_lexi(net, loader, optimizer, criterion, criterion_lexi, split, writer, args, train=True, plot=False, epoch=0):

    if train:
        net.img2state_net.eval()
        net.set_cls.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc="{1} E{0:02d}".format(epoch, "train" if train else "val "),
    )

    # Plot initial predictions in Tensorboard
    if plot and epoch == 0:
        utils.write_expls(net, loader, f"Expl/prior_{split}", 0, writer)

    running_loss = 0
    running_ra_loss = 0
    running_lexi_loss = 0
    preds_all = []
    labels_all = []
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):

        # input is either a set or an image
        imgs, target_set, img_class_ids, img_ids, _, table_expls = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr = net(imgs)
        # for training only set transformer given GT symbols
        # output_cls = net.set_cls(target_set)

        _, preds = torch.max(output_cls, 1)

        # convert sorting gt target set and gt table explanations to match the order of the predicted table
        target_set, match_ids = utils.hungarian_matching(output_attr.to('cuda'), target_set)
        table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]

        loss_pred = criterion(output_cls, img_class_ids)

        loss_lexi = criterion_lexi(net.set_cls, output_attr, img_class_ids, table_expls, epoch, batch_id=i,
                                   writer=None, writer_prefix=split)

        l2_grads = args.l2_grads
        loss = loss_pred + l2_grads * loss_lexi

        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_ra_loss += loss_pred.item()
        running_lexi_loss += loss_lexi.item()/ args.l2_grads

        labels_all.extend(img_class_ids.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        # Plot predictions in Tensorboard
        if plot and not(i % iters_per_epoch):
            utils.write_expls(net, loader, f"Expl/{split}", epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    if split == "train":
        writer.add_scalar(f"L2_grads/{split}_l2_grads", l2_grads, epoch)
    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Loss/{split}_ra_loss", running_ra_loss / len(loader), epoch)
    writer.add_scalar(f"Loss/{split}_lexi_loss", running_lexi_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("{} Loss: {:.4f}.. ".format(split, running_loss / len(loader)),
          "{} RA Loss: {:.4f}.. ".format(split, running_ra_loss / len(loader)),
          "{} Lexi Loss: {:.4f}.. ".format(split, running_lexi_loss / len(loader)),
          "{} Accuracy: {:.4f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


def train(args, task):

    task_dir = os.path.join(args.data_dir, task)

    if args.dataset == "clevr-hans-state":
        dataset_train = data.CLEVR_HANS_EXPL(
            task_dir, task, "train", lexi=True
        )
        dataset_val = data.CLEVR_HANS_EXPL(
            task_dir, task, "val", lexi=True
        )
        dataset_test = data.CLEVR_HANS_EXPL(
            task_dir, task, "test", lexi=True
        )
    else:
        print("Wrong dataset specifier")
        exit()

    args.n_imgclasses = dataset_train.n_classes
    args.class_weights = torch.ones(args.n_imgclasses)/args.n_imgclasses
    args.classes = np.arange(args.n_imgclasses)
    args.category_ids = dataset_train.category_ids

    train_loader = data.get_loader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_loader = data.get_loader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    if task == 'class_0':
        net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                                n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                                category_ids=args.category_ids, device=args.device)

        # load pretrained concept embedding module
        log = torch.load("logs/slot-attention-clevr-state-3_final", map_location=torch.device(args.device))
        net.img2state_net.load_state_dict(log['weights'], strict=True)
        print("Pretrained slot attention model loaded!")

        net = net.to(args.device)
    else:
        assert (task == 'class_1' or task == 'class_2')
        # load best model
        net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)
        net = net.to(args.device)

        temp_list = []
        checkpoint_filepath = './runs/IF_latest/concept-learner-xil-0-IF_latest_seed0/'
        for fname in os.listdir(checkpoint_filepath):
            if fname.endswith('.pth'):
                temp_list.append(fname)
        if len(temp_list) > 1:
            print("multiple paths to load from")
            temp_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) #get model w/best loss
            print(temp_list)
            time.sleep(2)
        checkpoint = torch.load(os.path.join(checkpoint_filepath, temp_list[-1]))

        net.load_state_dict(checkpoint['weights'])
        net.eval()
        print("\nModel loaded from checkpoint for next task\n")

    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(
        [p for name, p in net.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    criterion_lexi = LexiLoss(class_weights=args.class_weights.float().to("cuda"), args=args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    torch.backends.cudnn.benchmark = True

    # Create RTPT object
    rtpt = RTPT(name_initials='LB', experiment_name=f"IF_xil",
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # tensorboard writer
    writer = utils.create_writer(args)
    # writer = None

    cur_best_val_loss = np.inf
    TA, TL = [],[]
    VA, VL = [],[]
    TABLE_EXPLS_TOTAL = []
    early_stopper = EarlyStopper(patience=3)
    for epoch in range(args.epochs):
        train_acc, train_loss, table_ex = run_lexi(net, train_loader, optimizer, criterion, criterion_lexi, task, split='train', args=args,
                             writer=writer, train=True, plot=False, epoch=epoch)
        #test = torch.sum(input=table_ex, dim=0)
        #test = table_ex.detach().cpu().numpy()
        #print(test.shape)
        #utils.save_expls(net, train_loader, f'{task}', 'output/figures/')
        #utils.overlay_table_expl(test, 'output/figures/', f'{task}')
        #exit(1)
        TA.append(train_acc)
        TL.append(train_loss)

        # summed = torch.sum(table_ex, dim=1)
        # print(summed.shape)
        # TABLE_EXPLS_TOTAL.append(summed)
        scheduler.step()

    # result = torch.stack(TABLE_EXPLS_TOTAL, dim=0).sum(dim=0).sum(dim=0)
    # result = result.detach().cpu().numpy()
    # print("result", result.shape)
    # detach = [x.detach().cpu().numpy() for x in TABLE_EXPLS_TOTAL]
    # detach = torch.sum(detach, dim=0)


    # utils.overlay_table_expl(detach, 'output/figures/', f'{task}')
    # exit(1)

        val_acc, val_loss, _ = run_lexi(net, val_loader, optimizer, criterion, criterion_lexi, task, split='val', args=args,
                            writer=writer, train=False, plot=True, epoch=epoch)
        VA.append(val_acc)
        VL.append(val_loss)
        if early_stopper.early_stop(val_loss, VL):
            print("We are at epoch:", epoch)
            break
        _ = run(net, test_loader, optimizer, criterion, task, split='test', args=args, writer=writer,
                train=False, plot=False, epoch=epoch)

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": args,
        }
        if cur_best_val_loss > val_loss:
            if epoch > 0:
                # remove previous best model
                os.remove(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
            torch.save(results, os.path.join(writer.log_dir, "model_epoch{}_bestvalloss_{:.4f}.pth".format(epoch,
                                                                                                           val_loss)))
            cur_best_val_loss = val_loss

        # Update the RTPT (subtitle is optional)
        rtpt.step()

    # load best model for final evaluation
    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)
    net = net.to(args.device)
    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    get_confusion_from_ckpt(net, test_loader, criterion, args=args, task=task, datasplit='test_best',
                            writer=writer)
    get_confusion_from_ckpt(net, val_loader, criterion, args=args, task=task, datasplit='val_best',
                            writer=writer)

    # plot expls
    run(net, train_loader, optimizer, criterion, task, split='train_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    run(net, val_loader, optimizer, criterion, task, split='val_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    run(net, test_loader, optimizer, criterion, task, split='test_best', args=args, writer=writer, train=False, plot=True, epoch=0)

    writer.close()
    return VA, VL, TA, TL



def plot(args):

    print(f"\n\n{args.name} seed {args.seed}\n")

    # no positional info per object
    if args.dataset == "clevr-hans-state":
        dataset_val = data.CLEVR_HANS_EXPL(
            args.data_dir, "val", lexi=True, conf_vers=args.conf_version
        )
        dataset_test = data.CLEVR_HANS_EXPL(
            args.data_dir, "test", lexi=True, conf_vers=args.conf_version
        )
    else:
        print("Wrong dataset specifier")
        exit()

    args.n_imgclasses = dataset_val.n_classes
    args.class_weights = torch.ones(args.n_imgclasses)/args.n_imgclasses
    args.classes = np.arange(args.n_imgclasses)
    args.category_ids = dataset_val.category_ids

    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # load best model for final evaluation
    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)
    net = net.to(args.device)

    checkpoint = torch.load(args.fp_ckpt)
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    save_dir = args.fp_ckpt.split('model_epoch')[0]+'figures/'
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        # directory already exists
        pass

    # change plotting function in utils in order to visualize explanations
    assert args.conf_version == 'CLEVR-Hans3'
    utils.save_expls_attn(net, test_loader, "test", save_path=save_dir)


def generate_matrices(dataset_train, dataset_val, dataset_test, saving_route):
    args.n_imgclasses = dataset_train.n_classes
    args.class_weights = torch.ones(args.n_imgclasses)/args.n_imgclasses
    args.classes = np.arange(args.n_imgclasses)
    args.category_ids = dataset_train.category_ids

    train_loader = data.get_loader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,)
    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,)
    val_loader = data.get_loader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,)

    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                        n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                        category_ids=args.category_ids, device=args.device)

    # load pretrained concept embedding module
    log = torch.load("logs/slot-attention-clevr-state-3_final", map_location=torch.device(args.device))
    net.img2state_net.load_state_dict(log['weights'], strict=True)
    print("Pretrained slot attention model loaded!")

    net = net.to(args.device)

    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(
        [p for name, p in net.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion_lexi = LexiLoss(class_weights=args.class_weights.float().to("cuda"), args=args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    torch.backends.cudnn.benchmark = True

    # Create RTPT object
    rtpt = RTPT(name_initials='LB', experiment_name=f"IF_xil", max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # tensorboard writer
    writer = utils.create_newwriter(saving_route, args)
    # writer = None

    cur_best_val_loss = np.inf
    lexi_VL = []

    #early_stopper = EarlyStopper(patience=3)
    for epoch in range(args.epochs):
        _ = run_lexi(net, train_loader, optimizer, criterion, criterion_lexi, split='train', args=args,
                             writer=writer, train=True, plot=False, epoch=epoch)
        scheduler.step()
        val_loss = run_lexi(net, val_loader, optimizer, criterion, criterion_lexi, split='val', args=args,
                            writer=writer, train=False, plot=True, epoch=epoch)
        lexi_VL.append(val_loss)
        # if early_stopper.early_stop(val_loss, lexi_VL):
        #     print("We are at epoch:", epoch)
        #     break
        _ = run(net, test_loader, optimizer, criterion, split='test', args=args, writer=writer,
                train=False, plot=False, epoch=epoch)

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": args,
        }
        if cur_best_val_loss > val_loss:
            if epoch > 0:
                # remove previous best model
                os.remove(glob.glob(os.path.join(writer.log_dir, "*_model_*_bestvalloss*.pth"))[0])
            torch.save(results, os.path.join(writer.log_dir, "{}_model_epoch{}_bestvalloss_{:.4f}.pth".format(saving_route, epoch,
                                                                                                           val_loss)))
            cur_best_val_loss = val_loss

        # Update the RTPT (subtitle is optional)
        rtpt.step()

    # load best model for final evaluation
    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)
    net = net.to(args.device)
    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "*_model_*_bestvalloss*.pth"))[0])
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best',
                            writer=writer)
    get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best',
                            writer=writer)

    # plot expls
    TA, TL = [],[]

    _,_ = run(net, train_loader, optimizer, criterion, split='train_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    _, _ = run(net, val_loader, optimizer, criterion, split='val_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    testacc, testloss = run(net, test_loader, optimizer, criterion, split='test_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    
    TA.append(testacc)
    TL.append(testloss)

    writer.close()
    return TA, TL


if __name__ == "__main__":
    args = get_args()
    if args.mode == 'train':
        
        M_l = []
        M_a = []
        test_l = []

        #get dataloaders for each combi
        for train_task in ['class_0', 'class_1', 'class_2']:
            task_dir = os.path.join(args.data_dir, train_task)
            # train for each task
            print("train:", train_task)
            dataset_train = data.CLEVR_HANS_EXPL(
                task_dir, train_task, "train", lexi=True)

            for test_task in ['class_0', 'class_1', 'class_2']:
                task_dir = os.path.join(args.data_dir, test_task)
                saving_route = train_task + test_task
                # val and test for each task
                print("val/test:", test_task)
                dataset_val = data.CLEVR_HANS_EXPL(
                    task_dir, test_task, "val", lexi=True)
                dataset_test = data.CLEVR_HANS_EXPL(
                    task_dir, test_task, "test", lexi=True)

                TA, TL = generate_matrices(dataset_train, dataset_val, dataset_test, saving_route)
                final_acc = TA[-1]
                final_loss = TL[-1]
                M_a.append(final_acc)
                M_l.append(final_loss)
                test_l.append(saving_route)

        print("matrix acc", M_a, "matrix loss", M_l, "test", test_l)

        acc= np.asarray(M_a)
        new_acc = acc.reshape(3,3)
        output_acc = new_acc.T
        loss = np.asarray(M_l)
        new_loss = loss.reshape(3,3)
        output_loss = new_loss.T

        print(output_loss, output_acc)


        utils.plot_heatmaps(output_acc, output_loss)