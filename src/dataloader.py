import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
import glob
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

from pathlib import Path

from anomalib.data.btech import BTech
from anomalib.data.mvtec import MVTec
from anomalib.data.utils import InputNormalizationMethod

# required for certain images in OCT2017
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_btad(split, subset, batch_size=64, num_workers=8):
    dataset_root = Path.cwd().parent / "datasets" / "BTech"

    btech_datamodule = BTech(
        root=dataset_root,
        category=subset,
        image_size=256,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        task='classification',
        normalization=InputNormalizationMethod.NONE,  # don't apply normalization, as we want to visualize the images
    )
    # check if the data is available and setup
    btech_datamodule.prepare_data()
    btech_datamodule.setup()
    if split == 'train':
        return btech_datamodule.train_data
    elif split == 'test':
        return btech_datamodule.test_data
    elif split == 'val':
        return btech_datamodule.val_data
    else:
        raise NotImplementedError


def load_mvtec(split, subset, batch_size=64, num_workers=8):
    dataset_root = Path.cwd().parent / "datasets" / "MVTec"
    # MVTec Classification Train Set
    mvtec_datamodule = MVTec(
        root=dataset_root,
        category=subset,
        image_size=256,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        task="classification",
        normalization=InputNormalizationMethod.NONE,  # don't apply normalization, as we want to visualize the images
    )
    # check if the data is available and setup
    mvtec_datamodule.prepare_data()
    mvtec_datamodule.setup()
    if split == 'train':
        return mvtec_datamodule.train_data
    elif split == 'test':
        return mvtec_datamodule.test_data
    elif split == 'val':
        return mvtec_datamodule.val_data
    else:
        raise NotImplementedError
    
def load_miic(split):
    file_list, label_list = [], []
    data_dir = f'data/miic/{split}/'
    paths = []
    # only normal samples
    if split == 'train':
        path = (0, data_dir+'test_normal_*.jpg')
        paths.append(path)
    # only abnormal
    elif split == 'train_abnormal':
        data_dir = data_dir = f'data/miic/train/'
        path = (1, data_dir+'test_abnormal_*.jpg')
        paths.append(path)
    # abnormal and normal
    # ignore any masks or paddings
    elif split == 'test':
        path_normal = (0, data_dir+'*normal_[0-9][0-9][0-9][0-9][0-9].jpg')
        path_abnormal = (1, data_dir+'*abnormal_[0-9][0-9][0-9][0-9][0-9].jpg')
        paths.append(path_normal)
        paths.append(path_abnormal)
    for label, path in paths:
        for filename in glob.glob(path):
            file_list.append(filename)
            label_list.append(label)

    return file_list, label_list
class LoadDataset(Dataset):
    def __init__(self, data_dir, split, ext='jpeg', subset=None, batch_size=64, num_workers=8, anomalib_dataset=False):
        self.data_dir = data_dir
        self.ext = ext
        self.split = split
        self.num_workers = num_workers
    

        # relevant for BTAD and MVTEC datasets
        self.subset = subset
        self.batch_size = batch_size
        
        # relevant for fastflow
        self.anomalib_dataset = anomalib_dataset
        if (data_dir == 'btech') or (data_dir == 'mvtec'):
            self.preload_files()
            # preload relavant at the get_item function
            self.preload = True
        
        else:
            self.file_list, self.labels = self._get_file_list()
            self.preload = False

    def preload_files(self):
        if self.data_dir == 'btech':
            self.image_list = load_btad(self.split, self.subset, batch_size=self.batch_size, num_workers=self.num_workers)
        elif self.data_dir == 'mvtec':
            self.image_list = load_mvtec(self.split, self.subset, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            image_list = []
            for label, path in zip(self.labels, self.file_list):
                img = Image.open(path)
                img = preprocess_img(img)
                image_list.append((img, label))
            self.image_list = image_list

    def _get_file_list(self):
        file_list = []
        label_list = []

        if self.data_dir == 'miic':
            return load_miic(self.split)
        # set split to 'train'
        if self.split == 'train_abnormal':
            split = 'train'
        else:
            split = self.split

        if self.data_dir == 'OCT2017':
            path0 = (0, 'data/'+self.data_dir+'/'+split+'/NORMAL/')
            path1 = (1, 'data/'+self.data_dir+'/'+split+'/CNV/')
            path2 = (1, 'data/'+self.data_dir+'/'+split+'/DME/')
            path3 = (1, 'data/'+self.data_dir+'/'+split+'/DRUSEN/')
            paths = [path0, path1, path2, path3]
        elif self.data_dir == 'chest_xray':
            path0 = (0, 'data/'+self.data_dir+'/'+split+'/NORMAL/')
            path1 = (1, 'data/'+self.data_dir+'/'+split+'/PNEUMONIA/')
            paths = [path0, path1]

        if self.data_dir == 'mvtec':
            file_list, label_list = load_mvtec(self.split, subset=self.subset, batch_size=self.batch_size)
        elif self.data_dir == 'btech':
            file_list, label_list = load_btad(self.split, subset=self.subset, batch_size=self.batch_size)
        else:
            # OCT or chest_xray
            if self.split == 'train':
                paths = [path0]
            elif self.split == 'train_abnormal':
                if self.data_dir == 'OCT2017':
                    paths = [path1, path2, path3]
                else:
                    paths = [path1]
            for label, path in paths:
                for filename in glob.glob(path+'*.'+self.ext):
                    file_list.append(filename)
                    label_list.append(label)
        return file_list, label_list

    def __getitem__(self, index):
        if self.preload:
            if self.anomalib_dataset:
                data = {'image':self.image_list[index]['image'], 'label':self.image_list[index]['label']}
                return data
            else:
                img = self.image_list[index][0]
                label = self.image_list[index][1]

            # an extra check for when loading in data with anomalib
            # we need to make sure the train set only contains normal samples
            if (self.split == 'train') and (label != 0):
                raise ValueError('Train contains non-zero labels. The train set should not contain any samples considered abnormal.')
        else:
            file_path = self.file_list[index]
            img = Image.open(file_path)
            img = preprocess_img(img)
            label = self.labels[index]
        return img, label
    
    def __len__(self):
        if self.preload:
            return len(self.image_list) 
        return len(self.file_list)

def preprocess_img(img):
    convert_tensor = transforms.ToTensor()

    ## Resize every image as specified in the paper
    resizer = transforms.Resize((256,256))

    #Normalized using the regular ImageNet values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    apply_greyscale = transforms.Grayscale(num_output_channels=3)
    img = resizer(img)
    img = apply_greyscale(img)
    center_crop = transforms.CenterCrop(224)
    img = center_crop(img)
    img = convert_tensor(img)
    img = normalize(img)
    return img

def load(data_dir,batch_size=64, num_workers=4, subset=None, anomalib_dataset=False):
    train_dataset = LoadDataset(data_dir, split='train', subset=subset, num_workers=num_workers, batch_size=batch_size, anomalib_dataset=anomalib_dataset)
    test_dataset = LoadDataset(data_dir, split='test', subset=subset, num_workers=num_workers, batch_size=batch_size, anomalib_dataset=anomalib_dataset)

    # only the test set is loaded into the dataloader
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=False)
   

    if data_dir in ['btech', 'mvtec']:
        train_abnormal = LoadDataset(data_dir, split='val', subset=subset, batch_size=batch_size, anomalib_dataset=anomalib_dataset)
        return train_dataset, train_abnormal, test_loader
    else:
        train_abnormal = LoadDataset(data_dir, split='train_abnormal', batch_size=batch_size, subset=subset, anomalib_dataset=anomalib_dataset)
        return train_dataset, train_abnormal, test_loader
    
def split_data(n_splits, normal_data, abnormal_data):
    kfold_normal, kfold_abormal = KFold(n_splits, shuffle=True), KFold(n_splits, shuffle=True)
    normal_split, abnormal_split = list(kfold_normal.split(normal_data)), list(kfold_abormal.split(abnormal_data))
    train_split_normal = [kfold[0] for kfold in normal_split]
    test_split_normal = [kfold[1] for kfold in normal_split]
    train_split_abnormal = [kfold[0] for kfold in abnormal_split]
    test_split_abnormal = [kfold[1] for kfold in abnormal_split]
    return train_split_normal, test_split_normal, train_split_abnormal, test_split_abnormal

def fold_to_loaders(fold, train_split_normal, test_split_normal, train_split_abnormal, test_split_abnormal, n_workers, train_loader, train_abnormal):
    train_ids_normal, train_ids_abnormal = train_split_normal[fold], train_split_abnormal[fold]
    test_ids_normal, test_ids_abnormal = test_split_normal[fold], test_split_abnormal[fold]

    train_normal_dataset = torch.utils.data.dataset.Subset(train_loader,train_ids_normal)
    train_normal_loader = data.DataLoader(train_normal_dataset, num_workers = n_workers, shuffle=True, batch_size=64)
    train_abnormal_dataset =  torch.utils.data.dataset.Subset(train_abnormal,train_ids_abnormal)
    print(f"Number of abnormal vs normal samples in the threshold set: {len(train_abnormal_dataset.dataset)} vs {len(train_normal_dataset.dataset)}")
    threshold_dataset = torch.utils.data.ConcatDataset([train_abnormal_dataset, train_normal_dataset])
    threshold_loader = data.DataLoader(threshold_dataset, num_workers = n_workers, batch_size=64)

    validate_loader_normal = torch.utils.data.dataset.Subset(train_loader,test_ids_normal)
    validate_loader_abnormal = torch.utils.data.dataset.Subset(train_abnormal,test_ids_abnormal)
    validate_loader_combined = torch.utils.data.ConcatDataset([validate_loader_normal, validate_loader_abnormal])
    validate_loader_combined = data.DataLoader(validate_loader_combined, num_workers = n_workers, batch_size=64)
    return train_normal_loader, threshold_loader, validate_loader_combined