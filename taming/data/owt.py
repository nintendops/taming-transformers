import os
import numpy as np
import albumentations
import glob
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from PIL import Image


class OWTBase(Dataset):
    def __init__(self, size=None, dataroot="", onehot_segmentation=False, 
                 crop_size=None, force_no_crop=False, given_files=None):
        self.split = 'train' # self.get_split()
        self.size = size
        if crop_size is None:
            self.crop_size = size
        else:
            self.crop_size = crop_size

        self.onehot = onehot_segmentation       # return segmentation as rgb or one hot

        # file paths without extensions
        ids = [f[:-4].split('/')[-1] for f in glob.glob(os.path.join(dataroot, "*.JPG"))] # self.json_data["images"]     
        ids = ids*20

        self.labels = {"image_ids": ids}
        # self.img_id_to_captions = dict()
        self.img_id_to_filepath = dict()
        self.img_id_to_segmentation_filepath = dict()

        for iid in tqdm(ids, desc='ImgToPath'):
            self.img_id_to_filepath[iid] =  os.path.join(dataroot, iid+'.JPG')
            self.img_id_to_segmentation_filepath[iid] =  os.path.join(dataroot, iid+'.npy')

        # for imgdir in tqdm(imagedirs, desc="ImgToPath"):
        #     self.img_id_to_filepath[imgdir["id"]] = os.path.join(dataroot, imgdir["file_name"])
        #     self.img_id_to_captions[imgdir["id"]] = list()
           
        #     self.img_id_to_segmentation_filepath[imgdir["id"]] = os.path.join(
        #         self.segmentation_prefix, pngfilename)
          
        #     if given_files is not None:
        #         if pngfilename in given_files:
        #             self.labels["image_ids"].append(imgdir["id"])
        #     else:
        #         self.labels["image_ids"].append(imgdir["id"])

        # capdirs = self.json_data["annotations"]
        # for capdir in tqdm(capdirs, desc="ImgToCaptions"):
        #     # there are in average 5 captions per image
        #     self.img_id_to_captions[capdir["image_id"]].append(np.array([capdir["caption"]]))

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        if self.split=="validation":
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
        self.preprocessor = albumentations.Compose(
            [self.rescaler, self.cropper],
            additional_targets={"segmentation": "image"})
        if force_no_crop:
            self.rescaler = albumentations.Resize(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler],
                additional_targets={"segmentation": "image"})

    def __len__(self):
        return len(self.labels["image_ids"])

    def preprocess_image(self, image_path, segmentation_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # segmentation = Image.open(segmentation_path)
        # if not self.onehot and not segmentation.mode == "RGB":
        #     segmentation = segmentation.convert("RGB")

        segmentation = np.load(segmentation_path)
        # remove -1 labels
        segmentation = segmentation * (segmentation>-1)
        segmentation = segmentation.astype(np.uint8)

        # if self.onehot:
        #     assert self.stuffthing
        #     # stored in caffe format: unlabeled==255. stuff and thing from
        #     # 0-181. to be compatible with the labels in
        #     # https://github.com/nightrome/cocostuff/blob/master/labels.txt
        #     # we shift stuffthing one to the right and put unlabeled in zero
        #     # as long as segmentation is uint8 shifting to right handles the
        #     # latter too
        #     assert segmentation.dtype == np.uint8
        #     segmentation = segmentation + 1

        processed = self.preprocessor(image=image, segmentation=segmentation)
        image, segmentation = processed["image"], processed["segmentation"]
        image = (image / 127.5 - 1.0).astype(np.float32)

        if self.onehot:
            assert segmentation.dtype == np.uint8
            # make it one hot
            n_labels = 3
            flatseg = np.ravel(segmentation)
            onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)
            onehot[np.arange(flatseg.size), flatseg] = True
            onehot = onehot.reshape(segmentation.shape + (n_labels,)).astype(int)
            segmentation = onehot
        else:
            # normalizing to (-1, 1)?
            segmentation = (segmentation / 1.0 - 1.0).astype(np.float32)

        return image, segmentation

    def get(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
        image, segmentation = self.preprocess_image(img_path, seg_path)
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

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
        image, segmentation = self.preprocess_image(img_path, seg_path)
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


class OWTToken(OWTBase):
    def __init__(self, crop_size=None, dataroot="", dataset_name="", force_no_crop=False, given_files=None):
        
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
            self.img_id_to_segmentation_filepath[iid] =  os.path.join(dataroot, f'{iid}_cond.npy')

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


