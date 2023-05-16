import os
import json

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import numpy as np

from pycocotools import mask as coco_mask

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)

def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,)


CLASSES = {
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
}


class CLEVR_HANS_EXPL(torch.utils.data.Dataset):
    def __init__(self, base_path, task, split, lexi=False): #, conf_vers='conf_2'
        assert split in {
            "train",
            "val",
            "test",
        }
        self.lexi = lexi
        self.base_path = base_path
        self.task = task
        self.split = split
        self.max_objects = 10 #this should be equivalent to num of slots, OR ELSE
        #self.conf_vers = conf_vers

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]
        self.img_ids, self.img_class_ids, self.scenes, self.fnames, self.gt_img_expls, self.gt_table_expls = \
            self.prepare_scenes(scenes)

        self.transform = transforms.Compose(
            [transforms.Resize((128, 128)),
             transforms.ToTensor()]
        )
        # self.transform_img_expl = transforms.Compose(
        #     [transforms.ToPILImage(mode='L'),
        #      transforms.Resize((14, 14), interpolation=5),
        #      transforms.ToTensor()]
        # )

        self.n_classes = len(np.unique(self.img_class_ids))
        self.category_dict = CLASSES

        # get ids of category ranges, i.e. shape has three categories from ids 0 to 2
        self.category_ids = np.array([3, 6, 8, 10, 18])

    def convert_coords(self, obj, scene_directions):
        # convert the 3d coords based on camera position
        # conversion from ns-vqa paper, normalization for slot attention
        position = [np.dot(obj['3d_coords'], scene_directions['right']),
                    np.dot(obj['3d_coords'], scene_directions['front']),
                    obj['3d_coords'][2]]
        coords = [(p +4.)/ 8. for p in position]
        return coords

    def object_to_fv(self, obj, scene_directions):
        coords = self.convert_coords(obj, scene_directions)
        one_hot = lambda key: [obj[key] == x for x in CLASSES[key]]
        material = one_hot("material")
        color = one_hot("color")
        shape = one_hot("shape")
        size = one_hot("size")
        assert sum(material) == 1
        assert sum(color) == 1
        assert sum(shape) == 1
        assert sum(size) == 1
        # concatenate all the classes
        return coords + shape + size + material + color

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        gt_img_expls = []
        img_class_ids = []
        #task_ids = []
        gt_table_expls = []
        fnames = []
        for scene in scenes_json:
            fnames.append(os.path.join(self.images_folder, scene['image_filename']))

            #convert str class_id to int
            c = scene['image_filename'][15]
            if c == 'T':
                c_id = 1
            else:
                c_id = 0

            img_class_ids.append(c_id) 
            img_idx = scene["image_index"]

            # don't need task_id atm
            #task_ids.append(scene['class_id'])

            objects = [self.object_to_fv(obj, scene['directions']) for obj in scene["objects"]]
            objects = torch.FloatTensor(objects).transpose(0, 1)

            # get gt image explanation based on the classification rule of the class label
            gt_img_expl_mask = self.get_img_expl_mask(scene)
            gt_img_expls.append(gt_img_expl_mask)

            num_objects = objects.size(1)
            # pad with 0s
            if num_objects < self.max_objects:
                objects = torch.cat(
                    [
                        objects,
                        torch.zeros(objects.size(0), self.max_objects - num_objects),
                    ],
                    dim=1,
                )

            # get gt table explanation based on the classification rule of the class label
            gt_table_expl_mask = self.get_table_expl_mask(objects, scene['class_id'], c_id) #scene['class_id'] = task_id, class_id(T=1 or F=0)
            gt_table_expls.append(gt_table_expl_mask)

            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:num_objects] = 1

            # concatenate obj indication to end of object list
            objects = torch.cat((mask.unsqueeze(dim=0), objects), dim=0)

            img_ids.append(img_idx)
            scenes.append(objects.T)
        return img_ids, img_class_ids, scenes, fnames, gt_img_expls, gt_table_expls

    def get_img_expl_mask(self, scene):
        task_id = scene['class_id']
        class_id = scene['image_filename'][15]

        mask = 0

        for obj in scene['objects']:
            #should i change this to only task_id?
            if (task_id == 0 and class_id == 'T'):
                if (obj['shape'] == 'sphere' and obj['material'] == 'metal') or \
                        (obj['shape'] == 'cube' and obj['size'] == 'small'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
            elif (task_id == 1 and class_id == 'T'):
                if (obj['shape'] == 'sphere' and obj['color'] == 'red') or \
                        (obj['shape'] == 'cube' and obj['size'] == 'small'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)
            elif (task_id == 2 and class_id == 'T'):
                if (obj['shape'] == 'sphere' and obj['material'] == 'metal') or \
                        (obj['shape'] == 'cube' and obj['size'] == 'small'):
                        rle = obj['mask']
                        mask += coco_mask.decode(rle)

        return torch.tensor(mask) * 255 # for PIL

    def get_table_expl_mask(self, objects, task_id, c_id):
        objects = objects.T

        mask = torch.zeros(objects.shape)

        for i, obj in enumerate(objects):
            if (task_id == 0 and c_id == 1):
                #if metal sphere
                if (obj[3:10] == torch.tensor([1, 0, 0, 0, 0, 0, 1])).all():
                    mask[i, 3:10] = torch.tensor([1, 0, 0, 0, 0, 0, 1])
                #or small cube
                elif (obj[3:8] == torch.tensor([0, 1, 0, 0, 1])).all():
                    mask[i, 3:8] = torch.tensor([0, 1, 0, 0, 1])
            elif (task_id == 0 and c_id == 0):
                #if metal sphere
                if (obj[3:10] == torch.tensor([1, 0, 0, 0, 0, 0, 1])).all():
                    mask[i, 3:10] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
                #or small cube
                elif (obj[3:8] == torch.tensor([0, 1, 0, 0, 1])).all():
                    mask[i, 3:8] = torch.tensor([0, 0, 0, 0, 0])

            ############################################

            elif (task_id == 1 and c_id == 1):
                #if red sphere
                if ((obj[3:8] == torch.tensor([1, 0, 0, 0, 0])).all()
                    and (obj[10:] == torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])).all()).all():
                    mask[i, 3:8] = torch.tensor([1, 0, 0, 0, 0])
                    mask[i, 10:] = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])
                #or small cube
                elif (obj[3:8] == torch.tensor([0, 1, 0, 0, 1])).all():
                    mask[i, 3:8] = torch.tensor([0, 1, 0, 0, 1])
            elif (task_id == 1 and c_id == 0):
                #if red sphere
                if ((obj[3:8] == torch.tensor([1, 0, 0, 0, 0])).all()
                    and (obj[10:] == torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])).all()).all():
                    mask[i, 3:8] = torch.tensor([1, 0, 0, 0, 0])
                    mask[i, 10:] = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])
                #or small cube
                elif (obj[3:8] == torch.tensor([0, 1, 0, 0, 1])).all():
                    mask[i, 3:8] = torch.tensor([0, 0, 0, 0, 0])

            #############################################
            elif (task_id == 2 and c_id == 1):
                #if metal sphere
                if (obj[3:10] == torch.tensor([1, 0, 0, 0, 0, 0, 1])).all():
                    mask[i, 3:10] = torch.tensor([1, 0, 0, 0, 0, 0, 1])
                #or small cube
                elif (obj[3:8] == torch.tensor([0, 1, 0, 0, 1])).all():
                    mask[i, 3:8] = torch.tensor([0, 1, 0, 0, 1])
            elif (task_id == 2 and c_id == 0):
                #if metal sphere
                if (obj[3:10] == torch.tensor([1, 0, 0, 0, 0, 0, 1])).all():
                    mask[i, 3:10] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
                #or small cube
                elif (obj[3:8] == torch.tensor([0, 1, 0, 0, 1])).all():
                    mask[i, 3:8] = torch.tensor([0, 0, 0, 0, 0])

        return mask
    
    @property
    def task_folder(self):
        return os.path.join(self.base_path)

    @property
    def images_folder(self):
        return os.path.join(self.task_folder, self.split, "images") 

    @property
    def scenes_path(self):
        return os.path.join(
            self.base_path, self.split, "thesis_scenes_{}_{}.json".format(self.split, self.task) 
        )

    def __getitem__(self, item):
        image_id = self.img_ids[item]

        image = pil_loader(self.fnames[item])
        # TODO: sofar only dummy
        img_expl = torch.tensor([0])

        if self.transform is not None:
            image = self.transform(image) # in range [0., 1.]
            image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
            # img_expl = self.transform_img_expl(img_expl)

        objects = self.scenes[item]
        table_expl = self.gt_table_expls[item]
        img_class_id = self.img_class_ids[item]

        # remove objects presence indicator from gt table
        objects = objects[:, 1:]

        return image, objects, img_class_id, image_id, img_expl, table_expl

    def __len__(self):
        return len(self.scenes)

