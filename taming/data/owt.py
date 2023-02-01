import os
import numpy as np
import albumentations
import glob
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from PIL import Image

DEBUG_MODE = False

class OWTBase(Dataset):
    def __init__(self, size=None, dataroot="", multiplier=20, onehot_segmentation=False, ignore_segmentation=False,
                 crop_size=None, force_no_crop=False, given_files=None, multiscale_factor=1.0, extension='JPG', split='train'):
        self.ext = extension
        self.split = split # self.get_split()
        self.size = size
        self.multiplier = multiplier
        self.ms_factor = multiscale_factor
        if crop_size is None:
            self.crop_size = size
        else:
            self.crop_size = crop_size
        self.onehot = onehot_segmentation       # return segmentation as rgb or one hot
        self.dataroot = dataroot
        self.ignore_segmentation = ignore_segmentation

        self.initialize_paths()
        self.initialize_processor(force_no_crop)

    def __len__(self):
        return len(self.labels["image_ids"])

    def initialize_paths(self):
        # file paths without extensions
        ids = [f[:-4].split('/')[-1] for f in glob.glob(os.path.join(self.dataroot, f"*.{self.ext}"))] # self.json_data["images"]     
        ids = ids*self.multiplier

        self.labels = {"image_ids": ids}
        # self.img_id_to_captions = dict()
        self.img_id_to_filepath = dict()
        self.img_id_to_segmentation_filepath = dict()
        for iid in tqdm(ids, desc='ImgToPath'):
            self.img_id_to_filepath[iid] =  os.path.join(self.dataroot, iid+f'.{self.ext}')
            self.img_id_to_segmentation_filepath[iid] =  os.path.join(self.dataroot, iid+'.npy')

    def initialize_processor(self, force_no_crop=False):
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        if self.split=="validation":
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            self.hflipper = albumentations.HorizontalFlip(p=0.0)
        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.hflipper = albumentations.HorizontalFlip(p=0.5)

        # self.vflipper = albumentations.VerticalFlip(p=0.5)

        self.preprocessor = albumentations.Compose(
            [self.rescaler, self.cropper, self.hflipper],
            additional_targets={"segmentation": "image"})
        if force_no_crop:
            self.rescaler = albumentations.Resize(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler],
                additional_targets={"segmentation": "image"})

        if self.ms_factor < 1.0:
            self.rescaler_2 = albumentations.Resize(height=int(self.ms_factor*self.size), width=int(self.ms_factor*self.size))

    def preprocess_image(self, image_path, segmentation_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # segmentation = Image.open(segmentation_path)
        # if not self.onehot and not segmentation.mode == "RGB":
        #     segmentation = segmentation.convert("RGB")

        if self.ignore_segmentation:
            segmentation = image
        else:
            segmentation = np.load(segmentation_path)
            # remove -1 labels
            segmentation = segmentation * (segmentation>-1)
            segmentation = segmentation.astype(np.uint8)

        processed = self.preprocessor(image=image, segmentation=segmentation)
        image, segmentation = processed["image"], processed["segmentation"]

        if self.ms_factor < 1.0:
            image_rescaled = self.rescaler_2(image=image)['image']

        image = (image / 127.5 - 1.0).astype(np.float32)

        if self.ms_factor < 1.0:
            image_rescaled = (image_rescaled / 127.5 - 1.0).astype(np.float32)
        else:
            image_rescaled = image

        if self.onehot:
            assert segmentation.dtype == np.uint8
            # make it one hot
            n_labels = 3
            flatseg = np.ravel(segmentation)
            onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)
            onehot[np.arange(flatseg.size), flatseg] = True
            onehot = onehot.reshape(segmentation.shape + (n_labels,)).astype(int)
            segmentation = onehot.astype(np.float32)
        else:
            # normalizing to (-1, 1)?
            segmentation = (segmentation / 1.0 - 1.0).astype(np.float32)

        return image, segmentation, image_rescaled

    def get(self, i):
        if DEBUG_MODE:
            i = 0
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
        image, segmentation, image_rescaled = self.preprocess_image(img_path, seg_path)
        # captions = self.img_id_to_captions[self.labels["image_ids"][i]]
        # randomly draw one of all available captions per image
        # caption = captions[np.random.randint(0, len(captions))]
        example = {"image": image,
                   "image_rescale": image_rescaled,
                   # "caption": [str(caption[0])],
                   "segmentation": segmentation,
                   "img_path": img_path,
                   "seg_path": seg_path,
                   "filename_": img_path.split(os.sep)[-1]
                    }
        return example

    def __getitem__(self, i):

        if DEBUG_MODE:
            i = 0 # np.random.randint(low=0,high=2) # 0

        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
        image, segmentation, image_rescaled = self.preprocess_image(img_path, seg_path)
        # captions = self.img_id_to_captions[self.labels["image_ids"][i]]
        # randomly draw one of all available captions per image
        # caption = captions[np.random.randint(0, len(captions))]
        example = {"image": image,
                   # "caption": [str(caption[0])],
                   "segmentation": segmentation,
                   "img_path": img_path,
                   "seg_path": seg_path,
                   "filename_": img_path.split(os.sep)[-1]
                    }
        return example

class AnyImageFolder(OWTBase):
    def initialize_paths(self):
        # requires self.dataroot to be a txt file that stores all the sub-dataset paths
        all_ids = dict()
        ids = [f[:-4].split('/')[-1] for f in glob.glob(os.path.join(dataroot, f"*.{self.ext}"))] # self.json_data["images"]     
        # self.img_id_to_captions = dict()
        self.img_id_to_filepath = dict()
        self.img_id_to_segmentation_filepath = dict()
        for iid in tqdm(ids, desc='ImgToPath'):
            self.img_id_to_filepath[iid] =  os.path.join(dataroot, dataset_name, f'{iid}.{self.ext}')
        # no segmentation path
        self.img_id_to_segmentation_filepath = self.img_id_to_filepath
        self.labels = {"image_ids": ids }
        self.ignore_segmentation = True

def parse_txtfile(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    paths =[p.split(',')[0] for p in lines]
    names = [p.split(',')[1] for p in lines]
    return paths, names

class OWTMulti(OWTBase):
    def initialize_paths(self):
        # requires self.dataroot to be a txt file that stores all the sub-dataset paths
        all_ids = dict()
        paths, names = parse_txtfile(self.dataroot)
        for p,n in zip(paths,names): 
            all_ids[n] = [f[:-4].split('/')[-1] for f in glob.glob(os.path.join(p, f"*.{self.ext}"))] 

        # self.img_id_to_captions = dict()
        self.img_id_to_filepath = dict()
        self.img_id_to_segmentation_filepath = dict()

        for p,n in zip(paths,names):
            for iid in tqdm(all_ids[n], desc=f'ImgToPath_{n}'):
                set_id = f"{n}_{iid}"
                self.img_id_to_filepath[set_id] =  os.path.join(p, iid+f'.{self.ext}')
                self.img_id_to_segmentation_filepath[set_id] =  os.path.join(p, iid+'.npy')

        ids = [k for k in self.img_id_to_filepath.keys()]
        ids = ids * self.multiplier

        self.labels = {"image_ids": ids }



class OWTToken(OWTBase):
    def __init__(self, crop_size=None, dataroot="", dataset_name="", force_no_crop=False, given_files=None):
        self.split = 'train'
        self.crop_size = crop_size 

        # file paths without extensions
        ids = [f[:-4].split('/')[-1] for f in glob.glob(os.path.join(dataroot, "*.JPG"))] # self.json_data["images"]     
        ids = ids*20

        self.labels = {"image_ids": ids}
        # self.img_id_to_captions = dict()
        self.img_id_to_filepath = dict()
        self.img_id_to_segmentation_filepath = dict()

        for iid in tqdm(ids, desc='ImgToPath'):
            self.img_id_to_filepath[iid] =  os.path.join(dataroot, dataset_name, f'{iid}_img.npy')
            self.img_id_to_segmentation_filepath[iid] =  os.path.join(dataroot, dataset_name, f'{iid}_cond.npy')

        if self.split=="validation":
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)

        self.preprocessor = albumentations.Compose(
            [self.cropper],
            additional_targets={"segmentation": "image"})

    def preprocess_image(self, image_path, segmentation_path):
        image = np.load(image_path)
        segmentation = np.load(segmentation_path)
        processed = self.preprocessor(image=image, segmentation=segmentation)
        image, segmentation = processed["image"], processed["segmentation"]
        return image, segmentation

# class CustomBase(Dataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.data = None

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         example = self.data[i]
#         return example



# class OWTTrain(OWTBase):
#     def __init__(self, size, training_images_list_file):
#         super().__init__()
#         with open(training_images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         self.data = ImagePaths(paths=paths, size=size, random_crop=False)


# class OWtTest(OWTBase):
#     def __init__(self, size, test_images_list_file):
#         super().__init__()
#         with open(test_images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         self.data = ImagePaths(paths=paths, size=size, random_crop=False)


