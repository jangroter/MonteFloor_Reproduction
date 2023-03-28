import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
import torchvision
import os
import cv2
import json

from data_utils.poly_utils import *
from data_utils.s3d_utils import *


class MFLoader(object):
    def __init__(self, batch_size, mode, generate_input_candidates=False):
        self.mode = mode
        self.seed = 8978
        np.random.seed(seed=self.seed)

        self.batch_size = batch_size


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Selected device is:', device)
        self.device = device

        if mode == 'train':
            self.dataset = self.create_dataset(mode, generate_input_candidates)
            self.augment = True

            self.data = DataLoader(self.dataset, self.batch_size,
                                   drop_last=True,
                                   shuffle=True,
                                   collate_fn=self.collate_fn)

            self.sample_n = len(self.dataset)

        elif mode == 'online_eval' or mode == 'test':
            self.dataset = self.create_dataset(mode, generate_input_candidates)
            self.augment = False
            # self.batch_size = 4

            self.sample_n = len(self.dataset)

            self.data = DataLoader(self.dataset, self.batch_size,
                                   drop_last=True,
                                   collate_fn=self.collate_fn)


        elif mode == 'test':
            self.dataset = self.create_dataset( mode)
            self.augment = False
            self.batch_size = 1

            self.sample_n = 20

            self.data = DataLoader(self.dataset, self.batch_size,
                                   num_workers=1,
                                   drop_last=True,
                                   collate_fn=self.collate_fn)

        # elif mode == 'test':
        #     self.dataset = self.create_dataset(args, mode)
        #     self.augment = False
        #
        #     self.data = DataLoader(self.dataset,
        #                            1,
        #                            shuffle=False,
        #                            num_workers=1)
        #     self.sample_n = 20

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

    def collate_fn(self, samples):
        # images = []
        # masks = []
        # boxes = []
        # labels = []

        # for sample in samples:
        #     images.append(sample[0])
        #     boxes.append(sample[1]['boxes'])
        #     masks.append(sample[1]['masks'])
        #     labels.append(sample[1]['labels'])
        
        # images = [torch.as_tensor(i, dtype=torch.float32) for i in images]
        # masks = [torch.as_tensor(m, dtype=torch.uint8) for m in masks]
        # boxes = [torch.as_tensor(b, dtype=torch.float32) for b in boxes]
        # labels = [torch.as_tensor(l, dtype=torch.int64) for l in labels]
        
        # target = {}
        # target['masks'] = masks
        # target['boxes'] = boxes
        # target['labels'] = labels

        images = []
        targets = []
        
        for sample in samples:
            images.append(torch.tensor(sample[0], dtype=torch.float32).view(1,256,256))
            targets.append({
                'boxes': torch.tensor(sample[1]['boxes'], dtype=torch.float32),
                'labels': torch.tensor(sample[1]['labels'], dtype=torch.int64),
                'masks': torch.tensor(sample[1]['masks'], dtype=torch.uint8),
            })
        
        return images, targets


    def create_dataset(self, mode, generate_input_candidates):
        dataset_path = "montefloor_data"

        if mode == "train":
            scenes_path = os.path.join(dataset_path, "train")

            dataset = S3DDataset(scenes_path, None,
                                 num_scenes=3000, generate_input_candidates=generate_input_candidates, mode=mode)

        elif mode == "online_eval":
            scenes_path = os.path.join(dataset_path, "val")

            dataset = S3DDataset(scenes_path, None,
                                 num_scenes=250, generate_input_candidates=generate_input_candidates, mode=mode)
        elif mode == "test":
            scenes_path = os.path.join(dataset_path, "test")
            # scenes_path = os.path.join(dataset_path, "val")

            dataset = S3DDataset(scenes_path, None,
                                 num_scenes=250, generate_input_candidates=generate_input_candidates, mode=mode)

        return dataset

    def load_sample(self, sample_batch):
        """
        Identity function. Everything is already loaded in Dataset class for Structured 3D
        :param sample_batch:
        :return:
        """
        return sample_batch


class S3DDataset(Dataset):
    def __init__(self, scenes_path, score_gen, num_scenes, generate_input_candidates, mode):
        print("Creating Structured3D Dataset with %d scenes..." % num_scenes)
        self.score_gen = None

        self.mode = mode

        self.scenes_path = scenes_path
        self.floor_data_folder_name = ""

        self.scenes_list = os.listdir(scenes_path)
        self.scenes_list.sort()

        inv_scenes = ["scene_00756","scene_01155", "scene_01852", "scene_01192", "scene_01816"]
        self.scenes_list = [s for s in self.scenes_list if s not in inv_scenes]
        self.scenes_list = self.scenes_list[:(num_scenes-len(inv_scenes))]


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.gen_input_candidates = generate_input_candidates

    def __getitem__(self, item):
        scene_name = self.scenes_list[item]
        sample = self.load_scene(scene_name)
        image = sample['density_map']

        target = self.get_target(sample, item)


        return image, target

    def __len__(self):
        return len(self.scenes_list)
    
    def get_target(self,sample, item):
        """
        Create target dictionary with 'boxes', 'labels', and 'masks'
        """
        
        labels = self.get_labels(sample)
        masks = self.get_masks(sample)
        boxes = self.get_boxes(masks, item)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks

        return target
    
    def get_labels(self, sample):
        num_labels = int(np.max(sample['room_map']))
        labels = torch.ones(num_labels).type(torch.int64)
        return labels
        
    def get_masks(self, sample):
        num_masks = int(np.max(sample['room_map']))
        mask_map = torch.tensor(sample['room_map'])

        # Create a one-hot encoding of the mask_map
        one_hot_map = torch.nn.functional.one_hot(mask_map.long(), num_classes=num_masks+1)

        # Convert one-hot map to binary masks
        binary_masks = one_hot_map.permute(2, 0, 1).to(torch.uint8)

        return binary_masks[1:]

    def get_boxes(self, masks, item):
        try:
            boxes = torchvision.ops.masks_to_boxes(masks)
        except:
            print(item)
        return boxes


    def load_density_map(self, sp):
        """
        Load density map
        :param sp:
        :return:
        """
        density_path = os.path.join(sp, self.floor_data_folder_name, "density.png")
        density_map = cv2.imread(density_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 255.

        if self.gen_input_candidates:
            thresh = np.maximum(np.random.random(), 0.8)
            density_map = np.minimum(density_map, thresh) / thresh

        if self.mode != "test":
            pow = np.random.random()
            pow = (1.5 - 1.) * (pow - 1) + 1.5
            density_map = density_map ** pow


        return density_map.astype(np.float32)

    def load_annotation(self, sp):
        """
        Load annotation dict
        :param sp:
        :return:
        :rtype: dict
        """
        anno_path = os.path.join(sp, self.floor_data_folder_name, "annotation_3d.json")
        with open(anno_path, "r") as f:
            anno_dict = json.load(f)

        return anno_dict

    def load_scene(self, scene_name):
        """
        Load scene
        :param scene_name:
        :return:
        """

        def cvt_tmp_sample_to_torch():
            torch_sample = {}

            room_map = torch.tensor(np.array(sample['room_map']), device=self.device)[None]
            # room_map = kornia.morphology.dilation(room_map[:, None], kernel=torch.ones((5, 5), device=self.device))[:,0]

            torch_sample['room_map'] = room_map

            if 'input_map' in sample.keys():
                torch_sample['input_map'] = torch.tensor(np.array(sample['input_map']), device=self.device)[None]
                torch_sample['cand_inst'] = torch.tensor(np.array(sample['cand_inst']), device=self.device)[None]
                torch_sample['cand_confidence'] = torch.tensor(np.array(sample['cand_confidence']), device=self.device)[
                    None]

            else:
                torch_sample['density_map'] = torch.tensor(np.array(sample['density_map']), device=self.device)[None]
            torch_sample['wall_map'] = torch.tensor(np.array(sample['wall_map']), device=self.device)[None]
            # torch_sample['room_map'] = torch.tensor(np.array(sample['room_map']), device=self.device)[None]
            torch_sample['polygons_list'] = [torch.tensor(poly, device=self.device)[None] for poly in sample['polygons_list']]

            return torch_sample

        sp = os.path.join(self.scenes_path, scene_name)
        sample = {}
        sample["scene_name"] = scene_name

        scene_anno = self.load_annotation(sp)

        # density_map = torch.tensor(np.array(density_map))[None]
        density_map = self.load_density_map(sp)

        self.generate_room_map(sample, scene_anno, density_map)

        sample['density_map'] = density_map

        # import pdb; pdb.set_trace()
        for key, value in sample.items():
            assert np.all(value == value), "%s contains NaN" % key

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(131)
        # plt.title(scene_name)
        # plt.imshow(density_map)
        # plt.subplot(132)
        # plt.imshow(sample["room_map"])
        # plt.subplot(133)
        # # plt.imshow(sample["input_map"][:,:,1])
        # # plt.imshow(sample["cand_inst"][:,:,0])
        # plt.show()

        return sample

    def generate_room_map(self, sample, annos, density_map):
        """
        :param density_map:
        :param sample:
        :param annos:
        :return:
        """

        h, w = density_map.shape

        polys = parse_floor_plan_polys(annos)

        room_map, polygons_list = generate_floorplan(annos, polys, h, w, ignore_types=['outwall', 'door', 'window'], constant_color=False, shuffle=self.gen_input_candidates)

        room_map = cv2.dilate(room_map, np.ones((5,5)))


        wall_map, _ = generate_floorplan(annos, polys, h, w, ignore_types=[], include_types=['outwall'], constant_color=True)
        wall_map *= (room_map == 0)

        sample['room_map'] = room_map.astype(np.float32)
        sample['wall_map'] = wall_map.astype(np.float32)

        sample['polygons_list'] = polygons_list

    def generate_density(self, points, width=256, height=256):
        image_res_tensor = torch.tensor([width, height], device=self.device).reshape(1, 1, 2)

        coordinates = torch.round(points[:, :, :2] * image_res_tensor)
        coordinates = torch.minimum(torch.maximum(coordinates, torch.zeros_like(image_res_tensor)),
                                    image_res_tensor - 1).type(torch.cuda.LongTensor)

        density = torch.zeros((self.batch_size, height, width), dtype=torch.float, device=self.device)

        for i in range(self.batch_size):
            unique_coordinates, counts = torch.unique(coordinates[i], return_counts=True, dim=0)

            density[i, unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts.type(torch.cuda.FloatTensor)
            density[i] = density[i] / torch.max(density[i])

        return density