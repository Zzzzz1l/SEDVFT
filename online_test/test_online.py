import torch
import argparse
import pickle
import numpy as np
import itertools
import json
from dataset import COCO_Test

import sys

sys.path.append('../')
from data import TextField
from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention, \
    TransformerEnsemble

import random

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

from torch.utils.data import DataLoader

if __name__ == '__main__':

    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)

    # 测试集
    parser.add_argument('--features_path', type=str, default='/media/a1002/one/dataset/wyh/dataset/coco_all_align.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='annotations/captions_test2014.json')
    parser.add_argument('--image_ids_path', type=str, default='coco_test_image_id.json')
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    dataset = COCO_Test(feat_path=args.features_path, ann_file=args.annotation_folder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    encoder = TransformerEncoder(4, 0, attention_module=ScaledDotProductAttention)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 4, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    model_path = [
        'saved_models/ensemble_152_12_best_test.pth',
        'saved_models/ensemble_152_3_best_test.pth',
        'saved_models/ensemble_152_10_best_test.pth',
        'saved_models/ensemble_152_9_best_test_better.pth'
    ]
    ensemble_model = TransformerEnsemble(model=model, weight_files=model_path)

    outputs = []
    for it, (image_ids, images) in enumerate(iter(dataloader)):
        print('processing {} / {}'.format(it, len(dataloader.dataset) // args.batch_size))
        images = images.to(device)
        with torch.no_grad():
            out, _ = ensemble_model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        caps_gen = [' '.join([k for k, g in itertools.groupby(gen_i)]).strip() for gen_i in caps_gen]
        for i in range(image_ids.size(0)):
            item = {}
            item['image_id'] = int(image_ids[i])
            item['caption'] = caps_gen[i]
            outputs.append(item)

    output_path = 'results/captions_test2014_PGT_152_1_results.json'
    with open(output_path, 'w') as f:
        json.dump(outputs, f)

    print('finished!')