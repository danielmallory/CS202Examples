import os
import nltk
import torch
import pickle
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from build_vocab import Vocabulary
import torchvision.transforms as transforms


class CocoDataset(data.Dataset):
    # coco custom dataset compatible with torch.utils.data.DataLoader
    def __init__(self, root, json, vocab, transform=None):
        # set the path for images, captions, and vocabulary wrapper
        # args:
        #
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        # returns one data pair (image and caption)
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['ann_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # convert caption (string) to word ids
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)



def collate_fn(data):
    # creates mini-batch tensors from the list of tuples (image, caption)
    # we should build custom collate_fn rather than using default collate_fn
    # because merging caption (including padding) is not supported in default

    # args:
    # data: list of tuple (image, caption)
    #       - image: torch tensor of shape (3,256,256)
    #       - caption: torch tensor of shape k (var length for caption)

    # returns:
    #   images: torch tensor of shape (batch_size, 3, 256, 256)
    #   targets:  torch tensor of shape (batch_size, padded length)
    #   lengths: list; valid length for each padded caption

    # sort a data list by caption length (descending order)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # merge captions (from tuple of 1d tensor to 2d tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    # returns torch.utils.data.DataLoader for custom coco dataset

    # COCO caption dataset
    coco = CocoDataset(root=root, json=json, vocab=vocab, transform=transform)


    # data loader for coco dataset
    # this will return (images, captions, lengths) for each iteration
    # images: a tensor of shape (batch_size, 3, 224, 224)
    # captions: a tensor of shape (batch_size, padded_length)
    # lengths: a list indicating valid length for each caption. length is (batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)