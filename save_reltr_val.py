# Copyright (c) 2026 Nima Sharifinia
# Licensed under the Apache License, Version 2.0

path = "/home/nima/Downloads/"

import json
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
from torchvision import transforms
import re
from collections import Counter
import matplotlib.pyplot as plt

from torch_geometric.nn import GCNConv, GAT
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv  # Importing SAGEConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

class CustomTokenizer:
    def __init__(self, num_words=None, oov_token="<unk>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.max_length = 0
        self.word_to_idx = {}
        self.idx_to_word = {}

    def clean_text(self, text):
        # Convert to lowercase and remove punctuation (except spaces)
        text = text.lower()
        text = re.sub(r'[!"#$%&()*+.,-/:;=?@[\]^_`{|}~]', '', text)
        return text

    def fit_on_texts(self, texts):
        word_counts = {}

        # Split sentences into words correctly
        for text in texts:
            cleaned_text = self.clean_text(text)  # Proper cleaning
            words = cleaned_text.split()  # Split sentence into words
            if self.max_length < len(words):
              self.max_length = len(words)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort words by frequency
        sorted_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Limit vocab to num_words (excluding OOV)
        vocab_size = self.num_words - 1 if self.num_words else len(sorted_vocab)

        # Build the word_to_idx and idx_to_word dictionaries
        self.word_to_idx = {
            "<pad>": 0,
            "<unk>": 1,
        }
        self.idx_to_word = {
            0: "<pad>",
            1: "<unk>",
        }

        for idx, (word, _) in enumerate(sorted_vocab[:vocab_size], start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def text_to_sequence(self, text):
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()  # Ensure proper tokenization
        return [self.word_to_idx.get(word, self.word_to_idx["<unk>"]) for word in words]  # Map to index



class CocoValDataSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.all_captions = []
        self.all_images = []
        self.all_ids = []
        self.max_length = 0
        self.PATH = f"{path}/coco2017/val2017/"

        # Load annotations
        with open(f'{path}/coco2017/annotations/captions_val2017.json', 'r') as file:
            data = json.load(file)

        # Create an image index
        image_id_index = {}
        for img in data['images']:
            image_id_index[img['id']] = img['file_name']

        for annot in data['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            if not image_id in self.all_ids:
              full_coco_image_path = self.PATH + image_id_index[image_id]
              self.all_ids.append(image_id)
              self.all_images.append(full_coco_image_path)
              self.all_captions.append(caption)

    def encode(self, caption):
        caption = tokenizer.clean_text(caption)
        encoded_caption = tokenizer.text_to_sequence(caption)
        remain = tokenizer.max_length - len(encoded_caption)
        for i in range(remain):
            encoded_caption.append(0)

        return torch.tensor(encoded_caption)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.all_captions)

    def __getitem__(self, idx):
        """Fetches the image and one-hot encoded caption at the specified index."""
        image_name = self.all_images[idx]

        image = Image.open(image_name)
        image = self.transform(image)

        return image_name, image


# Example usage
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure 3 channels (convert grayscale to RGB)
    transforms.ToTensor(),
])

# Create the custom dataset
val_dataset = CocoValDataSet(transform=transform)
val_dataset_len = len(val_dataset)
tokenizer = CustomTokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(val_dataset.all_captions)

CLASSES = [ 'NA', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['background', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']


from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR

position_embedding = PositionEmbeddingSine(128, normalize=True)
backbone = Backbone('resnet50', False, False, False)
backbone = Joiner(backbone, position_embedding)
backbone.num_channels = 2048

transformer = Transformer(d_model=256, dropout=0.3, nhead=8,
                          dim_feedforward=2048,
                          num_encoder_layers=6,
                          num_decoder_layers=6,
                          normalize_before=False,
                          return_intermediate_dec=True)

rel_tr_model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
              num_entities=100, num_triplets=200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The checkpoint is pretrained on Visual Genome
ckpt = torch.load(
    'checkpoint_reltr.pth',
    map_location=device,
    weights_only=False)

for param in rel_tr_model.parameters():
    param.requires_grad = False

reltr_model = rel_tr_model.to(device)
reltr_model.load_state_dict(ckpt['model'])
reltr_model.eval()

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
          (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


coded_classes = []
coded_rels = []

for cls in CLASSES:
    cleaned_cls = tokenizer.clean_text(cls)
    words = cleaned_cls.split(" ")
    coded_words = []
    for w in words:
      coded_words.append(tokenizer.word_to_idx.get(w, tokenizer.word_to_idx["<unk>"]))
    coded_classes.append(coded_words)

max_len = 0
for cls in REL_CLASSES:
    cleaned_cls = tokenizer.clean_text(cls)
    words = cleaned_cls.split(" ")
    coded_words = []
    for w in words:
      coded_words.append(tokenizer.word_to_idx.get(w, tokenizer.word_to_idx["<unk>"]))
    coded_rels.append(coded_words)
    if len(words) > max_len:
      max_len = len(words)

# Max length of rel classes = max_len
# Now make the length of rel classes same as max_len (padding)
padded_tensors = []
for t in coded_rels:
    # اگر طول تنسور کمتر از max_len است، آن را با صفر پد می‌کنیم
    padding = max_len - len(t)
    for i in range(padding):
        t.append(0)
    padded_tensors.append(t)

padded_tensors = torch.tensor(padded_tensors)
coded_classes = torch.tensor(coded_classes)
coded_rels = torch.tensor(padded_tensors)


transform_reltr = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



class ModifiedReltr(torch.nn.Module):
  def __init__(self):
      super().__init__()
      self.rel_tr_model = rel_tr_model.to(device)
      self.rel_tr_model.load_state_dict(ckpt['model'])
      self.rel_tr_model.eval()

  def forward(self, x):
    outputs_all = []
    for im in x:
      img = transform_reltr(im).unsqueeze(0)
      # propagate through the model
      outputs = rel_tr_model(img)
      # keep only predictions with >0.3 confidence
      probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
      probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
      probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
      keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                              probas_obj.max(-1).values > 0.3))
      filtered_probas = probas[keep]
      filtered_probas_sub = probas_sub[keep]
      filtered_probas_obj = probas_obj[keep]

      nodes = []
      node_ids = []
      # return nodes
      edges = [[],[]]
      edge_attr = []
      for idx, _ in enumerate(filtered_probas):
        obj = coded_classes[filtered_probas_obj[idx].argmax()]
        obj_id = filtered_probas_obj[idx].argmax()
        if not obj_id in node_ids:
          nodes.append(obj)
          node_ids.append(obj_id)
        sub = coded_classes[filtered_probas_sub[idx].argmax()]
        sub_id = filtered_probas_sub[idx].argmax()
        if not sub_id in node_ids:
          nodes.append(sub)
          node_ids.append(sub_id)


        # print(CLASSES[filtered_probas_obj[idx].argmax()])
        idx_obj = node_ids.index(filtered_probas_obj[idx].argmax())
        idx_sub = node_ids.index(filtered_probas_sub[idx].argmax())
        edges[0].append(idx_sub)
        edges[1].append(idx_obj)
        edge_attr.append(padded_tensors[filtered_probas[idx].argmax()])

      if (len(nodes)>0):
        np_array1 = torch.stack(nodes)
        np_array1 = np_array1.to(device,dtype=torch.float)
        # print(np_array1.shape)
        x_np1 = np_array1.reshape((np_array1.shape[0],1))
      else :
        x_np1 = torch.tensor([[]]).to(device)
      edges_new = [torch.from_numpy(np.array(e)) for e in edges]
      np_array2 = torch.stack(edges_new)
      x_np2 = np_array2.to(device,dtype=int)



      if (len(nodes)>0):
        x_np3 = torch.stack(edge_attr).to(device)
      else :
        x_np3 = torch.tensor([]).to(device)


      graph_data = Data(x=x_np1, edge_index=x_np2,edge_attr=x_np3)
      outputs_all.append(graph_data)


    return outputs_all
  

class MainGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MainGCN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)  # Using GraphConv
        self.conv2 = GraphConv(hidden_dim, output_dim)  # Using GraphConv
        self.fc = nn.Linear(output_dim, 2048)

    def forward(self, ls):
      out = []
      for g in ls:
        # print(g)
        x , edge_index , edge_attr = g.x , g.edge_index,g.edge_attr
        if(x.size()==(1, 0)):
          out.append(torch.zeros((1,2048)).to(device))
          continue
        # x = self.conv1(x, edge_index,edge_attr)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.conv2(x, edge_index,edge_attr)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, g.batch)

        x = self.fc(x)

        out.append(x)
      return torch.stack(out)
    
class MainGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(MainGAT, self).__init__()
        self.att1 = GAT(input_dim, hidden_dim,num_layers)  # Using GraphConv
        self.att2 = GAT(hidden_dim, output_dim,num_layers)  # Using GraphConv
        self.fc = nn.Linear(output_dim, 2048)

    def forward(self, ls):
      out = []
      for g in ls:
        # print(g)
        x , edge_index , edge_attr = g.x , g.edge_index,g.edge_attr
        if(x.size()==(1, 0)):
          out.append(torch.zeros((1,2048)).to(device))
          continue
        # x = self.conv1(x, edge_index,edge_attr)
        x = self.att1(x, edge_index,edge_attr=edge_attr)
        x = F.relu(x)
        # x = self.conv2(x, edge_index,edge_attr)
        x = self.att2(x, edge_index,edge_attr=edge_attr)

        x = global_mean_pool(x, g.batch)

        x = self.fc(x)

        out.append(x)
      return torch.stack(out)
    
relTR_model = ModifiedReltr().to(device)
graph_encoder = MainGCN(input_dim=-1, hidden_dim=4, output_dim=2048).to(device)

for i, (img_name, img_tensor) in enumerate(val_dataset):
    print(f"{i}/{val_dataset_len}")
    graphs = relTR_model(torch.tensor(img_tensor).unsqueeze(0).to(device))
    graph_features = graph_encoder(graphs)
    graph_features = graph_features.detach().cpu().numpy()
    np.save(img_name + "_reltr_gcn", graph_features)