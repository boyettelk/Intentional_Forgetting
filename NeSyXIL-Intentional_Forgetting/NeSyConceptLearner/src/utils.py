import numpy as np
import random
import io
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
# from skimage import color
from sklearn import metrics
from matplotlib import rc
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from captum.attr import IntegratedGradients
import seaborn as sns

axislabel_fontsize = 8
ticklabel_fontsize = 8
titlelabel_fontsize = 8

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_tensor(input_tensors, h, w):
    input_tensors = torch.squeeze(input_tensors, 1)

    for i, img in enumerate(input_tensors):
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = transforms.Resize([h, w])(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        if i == 0:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    final_output = torch.unsqueeze(final_output, 1)
    return final_output


def norm_saliencies(saliencies):
    saliencies_norm = saliencies.clone()

    for i in range(saliencies.shape[0]):
        if len(torch.nonzero(saliencies[i], as_tuple=False)) == 0:
            saliencies_norm[i] = saliencies[i]
        else:
            saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                 (torch.max(saliencies[i]) - torch.min(saliencies[i]))

    return saliencies_norm


def generate_intgrad_captum_table(net, input, labels):
    labels = labels.to("cuda")
    explainer = IntegratedGradients(net)
    saliencies = explainer.attribute(input, target=labels)
    # remove negative attributions
    saliencies[saliencies < 0] = 0.
    # normalize each saliency map by its max
    for k, sal in enumerate(saliencies):
        saliencies[k] = sal/torch.max(sal)
    return norm_saliencies(saliencies)


def test_hungarian_matching(attrs=torch.tensor([[[0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]],
                                                [[0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]]]).type(torch.float),
                            pred_attrs=torch.tensor([[[0.01, 0.1, 0.2, 0.1, 0.2, 0.2, 0.01],
                                                      [0.1, 0.6, 0.8, 0., 0.4, 0.001, 0.9]],
                                                     [[0.01, 0.1, 0.2, 0.1, 0.2, 0.2, 0.01],
                                                      [0.1, 0.6, 0.8, 0., 0.4, 0.001, 0.9]]]).type(torch.float)):
    hungarian_matching(attrs, pred_attrs, verbose=1)


def hungarian_matching(attrs, preds_attrs, verbose=0):
    """
    Receives unordered predicted set and orders this to match the nearest GT set.
    :param attrs:
    :param preds_attrs:
    :param verbose:
    :return:
    """
    assert attrs.shape[1] == preds_attrs.shape[1]
    assert attrs.shape == preds_attrs.shape
    from scipy.optimize import linear_sum_assignment
    matched_preds_attrs = preds_attrs.clone()
    idx_map_ids = []
    for sample_id in range(attrs.shape[0]):
        # using euclidean distance
        cost_matrix = torch.cdist(attrs[sample_id], preds_attrs[sample_id]).detach().cpu()

        idx_mapping = linear_sum_assignment(cost_matrix)
        # convert to tuples of [(row_id, col_id)] of the cost matrix
        idx_mapping = [(idx_mapping[0][i], idx_mapping[1][i]) for i in range(len(idx_mapping[0]))]

        idx_map_ids.append([idx_mapping[i][1] for i in range(len(idx_mapping))])

        for i, (row_id, col_id) in enumerate(idx_mapping):
            matched_preds_attrs[sample_id, row_id, :] = preds_attrs[sample_id, col_id, :]
        if verbose:
            print('GT: {}'.format(attrs[sample_id]))
            print('Pred: {}'.format(preds_attrs[sample_id]))
            print('Cost Matrix: {}'.format(cost_matrix))
            print('idx mapping: {}'.format(idx_mapping))
            print('Matched Pred: {}'.format(matched_preds_attrs[sample_id]))
            print('\n')
            # exit()

    idx_map_ids = np.array(idx_map_ids)
    return matched_preds_attrs, idx_map_ids


def create_writer(args):
    writer = SummaryWriter(f"runs/{args.conf_version}/{args.name}_seed{args.seed}", purge_step=0)

    writer.add_scalar('Hyperparameters/learningrate', args.lr, 0)
    writer.add_scalar('Hyperparameters/num_epochs', args.epochs, 0)
    writer.add_scalar('Hyperparameters/batchsize', args.batch_size, 0)

    # store args as txt file
    with open(os.path.join(writer.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")
    return writer

def create_newwriter(saving_route, args):
    writer = SummaryWriter(f"runs/{args.conf_version}/{args.name}_seed{args.seed}/{saving_route}", purge_step=0)

    writer.add_scalar('Hyperparameters/learningrate', args.lr, 0)
    writer.add_scalar('Hyperparameters/num_epochs', args.epochs, 0)
    writer.add_scalar('Hyperparameters/batchsize', args.batch_size, 0)

    # store args as txt file
    with open(os.path.join(writer.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")
    return writer


def create_expl_images(img, pred_attrs, table_expl_attrs, img_expl, true_class_name, pred_class_name, xticklabels):
    """
    """
    assert pred_attrs.shape[0:2] == table_expl_attrs.shape[0:2]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title("Img")

    ax[1].imshow(pred_attrs, cmap='gray')
    ax[1].set_ylabel('Slot. ID', fontsize=axislabel_fontsize)
    ax[1].yaxis.set_label_coords(-0.1, 0.5)
    ax[1].set_yticks(np.arange(0, 11))
    ax[1].yaxis.set_tick_params(labelsize=axislabel_fontsize)
    ax[1].set_xlabel('Obj. Attr', fontsize=axislabel_fontsize)
    ax[1].set_xticks(range(len(xticklabels)))
    ax[1].set_xticklabels(xticklabels, rotation=90, fontsize=ticklabel_fontsize)
    ax[1].set_title("Pred Attr")

    ax[2].imshow(img_expl)
    ax[2].axis('off')
    ax[2].set_title("Img Expl")

    im = ax[3].imshow(table_expl_attrs)
    ax[3].set_yticks(np.arange(0, 11))
    ax[3].yaxis.set_tick_params(labelsize=axislabel_fontsize)
    ax[3].set_xlabel('Obj. Attr', fontsize=axislabel_fontsize)
    ax[3].set_xticks(range(len(xticklabels)))
    ax[3].set_xticklabels(xticklabels, rotation=90, fontsize=ticklabel_fontsize)
    ax[3].set_title("Table Expl")

    fig.suptitle(f"True Class: {true_class_name}; Pred Class: {pred_class_name}", fontsize=titlelabel_fontsize)

    return fig


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    # print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))
    print('Precision: {:.3f} Recall: {:.3f}, Accuracy: {:.3f}: ,f1_score: {:.3f}'.format(precision*100,recall*100,
                                                                                         accuracy*100,f1_score*100))
    return precision, recall, accuracy, f1_score


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None,
                          cmap=plt.cm.Blues, sFigName='confusion_matrix.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(sFigName)
    return ax


def write_expls(net, data_loader, tagname, epoch, writer):
    """
    Writes NeSy Concpet Learner explanations to tensorboard writer.
    """

    attr_labels = ['Sphere', 'Cube', 'Cylinder',
                   'Large', 'Small',
                   'Rubber', 'Metal',
                   'Cyan', 'Blue', 'Yellow', 'Purple', 'Red', 'Green', 'Gray', 'Brown']

    net.eval()

    for i, sample in enumerate(data_loader):
        # input is either a set or an image
        imgs, target_set, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr = net(imgs)
        _, preds = torch.max(output_cls, 1)

        # convert sorting gt target set and gt table explanations to match the order of the predicted table
        target_set, match_ids = hungarian_matching(output_attr.to('cuda'), target_set)
        # table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]

        # get explanations of set classifier
        table_saliencies = generate_intgrad_captum_table(net.set_cls, output_attr, preds)

        # get the ids of the two objects that receive the maximal importance, i.e. most important for the classification
        max_expl_obj_ids = table_saliencies.max(dim=2)[0].topk(2)[1]

        # get attention masks
        attns = net.img2state_net.slot_attention.attn
        # reshape attention masks to 2D
        attns = attns.reshape((attns.shape[0], attns.shape[1], int(np.sqrt(attns.shape[2])),
                               int(np.sqrt(attns.shape[2]))))

        # concatenate the visual explanation of the top two objects that are most important for the classification
        img_saliencies = torch.zeros(attns.shape[0], attns.shape[2], attns.shape[3])
        for obj_id in range(max_expl_obj_ids.shape[1]):
            img_saliencies += attns[range(attns.shape[0]), obj_id, :, :].detach().cpu()

        # upscale img_saliencies to orig img shape
        img_saliencies = resize_tensor(img_saliencies.cpu(), imgs.shape[2], imgs.shape[2]).squeeze(dim=1).cpu()

        for img_id, (img, gt_table, pred_table, table_expl, img_expl, true_label, pred_label, imgid) in enumerate(zip(
                imgs, target_set, output_attr, table_saliencies,
                img_saliencies, img_class_ids, preds,
                img_ids
        )):
            # unnormalize images
            img = img / 2. + 0.5  # Rescale to [0, 1].

            fig = create_expl_images(np.array(transforms.ToPILImage()(img.cpu()).convert("RGB")),
                                           pred_table.detach().cpu().numpy(),
                                           table_expl.detach().cpu().numpy(),
                                           img_expl.detach().cpu().numpy(),
                                           true_label, pred_label, attr_labels)
            writer.add_figure(f"{tagname}_{img_id}", fig, epoch)
            if img_id > 10:
                break

        break


def save_expls(net, data_loader, tagname, save_path, task):
    """
    Stores the explanation plots at the specified location.
    """

    xticklabels = ['Sphere', 'Cube', 'Cylinder',
                   'Large', 'Small',
                   'Rubber', 'Metal',
                   'Cyan', 'Blue', 'Yellow', 'Purple', 'Red', 'Green', 'Gray', 'Brown']

    net.eval()
    relevant_ids = random.sample(range(0, 149), 30) #there are 150 images in test folder
    #print(relevant_ids)
    test = []

    for i, sample in enumerate(data_loader):
        # input is either a set or an image
        imgs, target_set, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr = net(imgs)
        _, preds = torch.max(output_cls, 1)

        # # convert sorting gt target set and gt table explanations to match the order of the predicted table
        # target_set, match_ids = utils.hungarian_matching(output_attr.to('cuda'), target_set)
        # # table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]

        # get explanations of set classifier
        table_saliencies = generate_intgrad_captum_table(net.set_cls, output_attr, preds)
        # remove xyz coords from tables for conf_3
        output_attr = output_attr[:, :, 3:]
        table_saliencies = table_saliencies[:, :, 3:]

        #IF addition
        new = torch.sum(table_saliencies, dim=0)
        #print("new shape", new.shape)
        new = new.detach().cpu()


        # get the ids of the two objects that receive the maximal importance, i.e. most important for the classification
        k = min(k, table_saliencies.shape[1])  # Limit k to the number of objects
        max_expl_obj_ids = table_saliencies.max(dim=2)[0].topk(k)[1]



        #get the ids of all objects since we only have max 4 anyway...
        #max_expl_obj_ids = table_saliencies.max(dim=2)[0].argmax(dim=1)

        # get attention masks
        attns = net.img2state_net.slot_attention.attn
        # reshape attention masks to 2D
        attns = attns.reshape((attns.shape[0], attns.shape[1], int(np.sqrt(attns.shape[2])),
                               int(np.sqrt(attns.shape[2]))))

        # concatenate the visual explanation of the top two objects that are most important for the classification
        img_saliencies = torch.zeros(attns.shape[0], attns.shape[2], attns.shape[3])
        batch_size = attns.shape[0]
        # for i in range(max_expl_obj_ids.shape[1]):
        #     img_saliencies += attns[range(batch_size), max_expl_obj_ids[range(batch_size), i], :, :].detach().cpu()

        for obj_id in max_expl_obj_ids:
            img_saliencies += attns[range(attns.shape[0]), obj_id, :, :].detach().cpu()

        num_stored_imgs = 0
        

        for img_id, (img, gt_table, pred_table, table_expl, img_expl, true_label, pred_label, imgid) in enumerate(zip(imgs, target_set, output_attr.detach().cpu().numpy(),table_saliencies.detach().cpu().numpy(), img_saliencies.detach().cpu().numpy(),img_class_ids, preds, img_ids)):
            if (imgid in relevant_ids and true_label == 1):
                test.append(pred_table)
                #print("test", imgid)

                num_stored_imgs += 1
                
                # norm img expl to be between 0 and 255
                img_expl = (img_expl - np.min(img_expl))/(np.max(img_expl) - np.min(img_expl))
                # resize to img size
                img_expl = np.array(Image.fromarray(img_expl).resize((img.shape[1], img.shape[2]), resample=1))

                # unnormalize images
                img = img / 2. + 0.5  # Rescale to [0, 1].
                img = np.array(transforms.ToPILImage()(img.cpu()).convert("RGB"))

                np.save(f"{save_path}/{tagname}_{imgid}.npy", img)
                np.save(f"{save_path}/{tagname}_{imgid}_imgexpl.npy", img_expl)
                np.save(f"{save_path}/{tagname}_{imgid}_table.npy", pred_table)
                np.save(f"{save_path}/{tagname}_{imgid}_tableexpl.npy", table_expl)
                np.save(f"{save_path}/{tagname}_{imgid}_newthingy.npy", new)

                fig = create_expl_images(img, pred_table, new, img_expl,
                                        true_label, pred_label, xticklabels)
                plt.savefig(f"{save_path}/{tagname}_{imgid}.png")
                plt.close(fig)

                if num_stored_imgs == len(relevant_ids): 
                    print("here")
                    exit()

    return test


def og_save_expls(net, data_loader, tagname, save_path):
    """
    Stores the explanation plots at the specified location.
    """

    xticklabels = ['Sphere', 'Cube', 'Cylinder',
                   'Large', 'Small',
                   'Rubber', 'Metal',
                   'Cyan', 'Blue', 'Yellow', 'Purple', 'Red', 'Green', 'Gray', 'Brown']

    net.eval()

    for i, sample in enumerate(data_loader):
        print("step one")
        # input is either a set or an image
        imgs, target_set, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr = net(imgs)
        _, preds = torch.max(output_cls, 1)

        # # convert sorting gt target set and gt table explanations to match the order of the predicted table
        # target_set, match_ids = utils.hungarian_matching(output_attr.to('cuda'), target_set)
        # # table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]

        # get explanations of set classifier
        table_saliencies = generate_intgrad_captum_table(net.set_cls, output_attr, preds)
        # remove xyz coords from tables for conf_3
        output_attr = output_attr[:, :, 2:]
        table_saliencies = table_saliencies[:, :, 2:]

        # output_attr = output_attr[:, :, 3:]
        # table_saliencies = table_saliencies[:, :, 3:]

        # get the ids of the two objects that receive the maximal importance, i.e. most important for the classification
        max_expl_obj_ids = table_saliencies.max(dim=2)[0].topk(2)[1]

        # k = min(5, table_saliencies.shape[1])  # Limit k to the number of objects
        # print("this is k", k)
        # max_expl_obj_ids = table_saliencies.max(dim=2)[0].topk(k)[1]

        # get attention masks
        attns = net.img2state_net.slot_attention.attn
        # reshape attention masks to 2D
        attns = attns.reshape((attns.shape[0], attns.shape[1], int(np.sqrt(attns.shape[2])),
                               int(np.sqrt(attns.shape[2]))))

        # concatenate the visual explanation of the top two objects that are most important for the classification
        img_saliencies = torch.zeros(attns.shape[0], attns.shape[2], attns.shape[3])
        batch_size = attns.shape[0]
        for i in range(max_expl_obj_ids.shape[1]):
            img_saliencies += attns[range(batch_size), max_expl_obj_ids[range(batch_size), i], :, :].detach().cpu()


        # batch_size = attns.shape[0]
        # img_saliencies = torch.zeros((batch_size, attns.shape[2], attns.shape[3]), device=attns.device, dtype=torch.float32)

        # for i in range(max_expl_obj_ids.shape[1]):
        #     obj_ids = max_expl_obj_ids[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #     saliency = torch.gather(attns, 1, obj_ids.expand(-1, -1, attns.shape[2], attns.shape[3])).detach()
        #     img_saliencies = torch.cuda.FloatTensor(img_saliencies) + saliency.squeeze(1)

        # img_saliencies = img_saliencies.detach().cpu()



        num_stored_imgs = 0
        relevant_ids = [618, 154, 436, 244, 318, 85]

        for img_id, (img, gt_table, pred_table, table_expl, img_expl, true_label, pred_label, imgid) in enumerate(zip(
                imgs, target_set, output_attr.detach().cpu().numpy(),
                table_saliencies.detach().cpu().numpy(), img_saliencies.detach().cpu().numpy(),
                img_class_ids, preds, img_ids
        )):
            if imgid in relevant_ids:
                num_stored_imgs += 1
                # norm img expl to be between 0 and 255
                img_expl = (img_expl - np.min(img_expl))/(np.max(img_expl) - np.min(img_expl))
                # resize to img size
                img_expl = np.array(Image.fromarray(img_expl).resize((img.shape[1], img.shape[2]), resample=1))

                # unnormalize images
                img = img / 2. + 0.5  # Rescale to [0, 1].
                img = np.array(transforms.ToPILImage()(img.cpu()).convert("RGB"))

                np.save(f"{save_path}{tagname}_{imgid}.npy", img)
                np.save(f"{save_path}{tagname}_{imgid}_imgexpl.npy", img_expl)
                np.save(f"{save_path}{tagname}_{imgid}_table.npy", pred_table)
                np.save(f"{save_path}{tagname}_{imgid}_tableexpl.npy", table_expl)

                fig = create_expl_images(img, pred_table, table_expl, img_expl,
                                         true_label, pred_label, xticklabels)
                plt.savefig(f"{save_path}{tagname}_{imgid}.png")
                plt.close(fig)

                if num_stored_imgs == len(relevant_ids):
                    exit()

########## IF functions ##########
def save_plots(TA, VA, TL, VL, task):
    """
    Function to save the loss and accuracy plots to disk.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10,7))
    fig.suptitle(f"Accuracy and Loss for Task {task}")
    axes[0].plot(TA, color='green', linestyle='-', label='train accuracy')
    axes[0].plot(VA, color='blue', linestyle='-', label='validation accuracy')
    axes[0].set_ylabel('Accuracy')


    axes[1].plot(TL, color='orange', linestyle='-', label='train loss')
    axes[1].plot(VL, color='red', linestyle='-', label='validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')

    fig.legend()
    plt.savefig(f'./output/plot_task{task}.png')  # will now save all plots


def plot_heatmaps(acc, loss):
    plt.clf()
    if not os.path.exists('./output'):
        os.makedirs('./output')

    xticks = ['task0', 'task1', 'task2']
    yticks = ['task0', 'task1', 'task2']
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 12))
    a = sns.heatmap(acc, cmap='crest', annot=True, fmt='.2f', xticklabels=xticks, yticklabels=yticks, ax=ax1)
    a.set(xlabel ="Train", ylabel = "Validate", title ='Accuracy')
    b = sns.heatmap(loss, cmap='crest', annot=True, fmt='.2f', xticklabels=xticks, yticklabels=yticks, ax=ax2)
    b.set(xlabel ="Train", title ='Loss')
    plt.tight_layout()
    fig.savefig('./output/heatmaps.png', dpi=400)
    fig.clf()


def expl_test(net, output_attr, preds, imgs, target_set, img_class_ids, img_ids):
    """
    Stores the explanation plots at the specified location.
    """

    # get explanations of set classifier
    table_saliencies = generate_intgrad_captum_table(net.set_cls, output_attr, preds)
    # remove xyz coords from tables for conf_3
    output_attr = output_attr[:, :, 3:]
    table_saliencies = table_saliencies[:, :, 3:]

    # get the ids of the two objects that receive the maximal importance, i.e. most important for the classification
    max_expl_obj_ids = table_saliencies.max(dim=2)[0].topk(6)[1]

    # get attention masks
    attns = net.img2state_net.slot_attention.attn
    # reshape attention masks to 2D
    attns = attns.reshape((attns.shape[0], attns.shape[1], int(np.sqrt(attns.shape[2])),
                            int(np.sqrt(attns.shape[2]))))

    # concatenate the visual explanation of the top two objects that are most important for the classification
    img_saliencies = torch.zeros(attns.shape[0], attns.shape[2], attns.shape[3])
    batch_size = attns.shape[0]
    for i in range(max_expl_obj_ids.shape[1]):
        img_saliencies += attns[range(batch_size), max_expl_obj_ids[range(batch_size), i], :, :].detach().cpu()
    
    test_saliencies = img_saliencies.detach().clone()

    num_stored_imgs = 0
    #relevant_ids = [618, 154, 436, 244, 318, 85]
    concat_table_expl = []

    for img_id, (img, gt_table, pred_table, table_expl, img_expl, true_label, pred_label, imgid) in enumerate(zip(
            imgs, target_set, output_attr.detach().cpu().numpy(),
            table_saliencies.detach().cpu().numpy(), img_saliencies.detach().cpu().numpy(),
            img_class_ids, preds, img_ids
    )):
        #if imgid in relevant_ids:
        for i in range(5):
            if true_label == 1:

                # collect table_expl for concat explanation
                concat_table_expl.concat(table_expl)
            
    return concat_table_expl

def part_two(input, tagname, save_path):
    xticklabels = ['Sphere', 'Cube', 'Cylinder',
                   'Large', 'Small',
                   'Rubber', 'Metal',
                   'Cyan', 'Blue', 'Yellow', 'Purple', 'Red', 'Green', 'Gray', 'Brown']
    plt.figure(figsize=(6,6))
    plt.imshow(input)
    # for table_expl in concat_table_expl:
    #     plt.imshow(table_expl)
    plt.yticks(ticks=np.arange(0, 11))
    #plt.yaxis.set_tick_params(labelsize=axislabel_fontsize)
    plt.xlabel('Obj. Attr', fontsize=axislabel_fontsize)
    plt.xticks(ticks=range(len(xticklabels)), labels=xticklabels, rotation=90)
    #plt.set_xticklabels(xticklabels, rotation=90, fontsize=ticklabel_fontsize)
    plt.title("Table Expl")

    plt.savefig(f"{save_path}{tagname}_concat.png")
    plt.clf()
