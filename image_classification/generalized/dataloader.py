import os
import json

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageClassificationHelper(Dataset):
    def __init__(self, data_dir:str, labels_dict:dict) -> None:
        """ Class to read image classification directory.

            Note: Arrange images in data_dir/<train/test>/<label_name>/*

        Args:
            data_dir (str): Directory where data is stored
            labels_dict (dict): Dictionary containing labels
        """
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.labels_dict = labels_dict
        self.train_files, self.train_labels = self.get_images(split='train')
        self.test_files, self.test_labels = self.get_images(split='test')

    def __len__(self, split='train'):
        return len(self.train_files) if split == 'train' else len(self.test_files)

    def get_images(self, split='train'):
        files_list, labels_list = [], []
        dir = self.train_dir if split == 'train' else self.test_dir
        labels = self.labels_dict

        for i, label in labels.items():
            label_dir = os.path.join(dir, label)

            for dir_path, dir_name, file_paths in os.walk(label_dir):
                for file_path in file_paths:
                    if os.path.splitext(os.path.basename(file_path))[1] in [".jpg", ".jpeg", ".png"]:
                        files_list.append(os.path.join(dir_path, file_path))
                        labels_list.append(int(i))
        return files_list, labels_list

    def create_datamap(self, split='train') -> pd.DataFrame:
        """Method to generate datamap for the dataloader.

        Args:
            split (str, optional): Either of 'train' or 'test'. Defaults to 'train'.

        Returns:
            pd.DataFrame: Dataframe containing index, filepaths and labels
        """
        files = self.train_files if split == 'train' else self.test_files
        labels = self.train_labels if split == 'train' else self.test_labels

        datamap = dict(index=range(len(files)), filepath=files, label=labels)
        datamap_df = pd.DataFrame(datamap)
        return datamap_df

    @staticmethod
    def update_datamap(original_datamap:pd.DataFrame, new_datamap:pd.DataFrame) -> pd.DataFrame:
        """Method to update datamap based on original datamap and new datamap. If records are deleted, exception is raised. New records are appended in the end to make sure no-reranking happens.

        Args:
            original_datamap (pd.DataFrame): Original datamap
            new_datamap (pd.DataFrame): New datamap containing new records added.

        Raises:
            Exception: If records are deleted from original datamap.

        Returns:
            pd.DataFrame: Updated datamap with new records appended in the end.
        """
        if original_datamap is None:
            return new_datamap

        # Validate all records in original_datamap with new_datamap
        validated_indices, deleted_indices = [], []
        for i in range(original_datamap.shape[0]):
            org_filename = original_datamap.loc[i, 'filepath']
            new_rec = new_datamap[new_datamap['filepath'] == org_filename]
            if new_rec.empty:
                deleted_indices.append(i)
                continue
            assert new_rec.shape[0] == 1, f"Found multiple new records for same filename: {new_rec['filename']}"
            new_rec = new_rec.iloc[0]

            validated_indices.append(new_rec['index'])

        # At this point, all original records have been validated, validate new records
        max_idx = original_datamap['index'].max()
        new_rec_idx = set(new_datamap['index']) - set(validated_indices)
        for i in deleted_indices:
            print(f"Deleted record: {i} | {original_datamap.loc[i, 'filepath']}")
            raise Exception(f"Found deleted record: {i} | {original_datamap.loc[i, 'filepath']}")

        new_rec_datamap = new_datamap.loc[list(new_rec_idx)]
        new_rec_idx = list(range(max_idx + 1, max_idx + 1 + len(new_rec_idx)))
        new_rec_dict = dict(index=new_rec_idx, filepath=new_rec_datamap['filepath'].values.tolist(), label=new_rec_datamap['label'].values.tolist())
        new_rec_datamap = pd.DataFrame(new_rec_dict)
        updated_datamap = pd.concat([original_datamap, new_rec_datamap], ignore_index=True)

        print(f"Dataset validated. Found {len(deleted_indices)} deleted records and {len(new_rec_idx)} new records.")
        return updated_datamap, new_rec_idx, new_rec_idx

class ImageClassificationDataloader:
    def __init__(self, datamap: pd.DataFrame, labeled:list=None, transform=None) -> None:
        """ Dataloader class to load images and labels based on the datamap with indices in labeled list. Apply transforms also if passed.

        Args:
            datamap (pd.DataFrame): Datamap containing index, filepath and label.
            labeled (list, optional): List of indices to select. Defaults to None.
            transform (optional): Pytorch transform. Defaults to None.
        """
        self.datamap = datamap
        if labeled is not None:
            assert self.datamap.shape[0] >= len(labeled), f"Not enough samples as per labeled list: Found samples: {self.datamap.shape[0]} | Got: {len(labeled)}"
            self.datamap = self.datamap.loc[labeled]
            self.datamap.reset_index(inplace=True)
        self.labeled = labeled
        self.transform = transform

    def __len__(self):
        return self.datamap.shape[0]

    def __getitem__(self, idx):
        if idx >= self.datamap.shape[0]:
            raise Exception(f"Following index not found in datamap: {idx}")

        image_filename = self.datamap.loc[idx]['filepath']
        label = self.datamap.loc[idx]['label']

        img = Image.open(image_filename)
        img = img.convert('RGB')
        # img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

if __name__ == '__main__':

    dataset_path = "/home/dell/Documents/personal_repos/AlectioExamples/image_classification/weather/data"
    with open('labels.json', 'r') as f:
        labels_dict = json.load(f)
    img_cls_obj = ImageClassificationHelper(dataset_path, labels_dict)
    datamap_df = img_cls_obj.create_datamap()

    org_datamap = pd.read_csv('datamap.csv')
    updated_datamap, new_rec_idx, _ = img_cls_obj.update_datamap(org_datamap, datamap_df)

    dataloader = ImageClassificationDataloader(datamap_df)
    for img, label in dataloader:
        print(f"Image shape: {img.size} | Label: {label}")
