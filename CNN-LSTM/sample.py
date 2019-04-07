import os
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
import torch.cuda as cuda
import matplotlib.pyplot as plt
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN

# device configuration
device = cuda.device('cuda' if cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def main(args):
    # image preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),
                                                                                (0.229, 0.224, 0.225))])

    # load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode: batch norm uses moving mean/variance
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    # if we have a GPU, run our computations on that, otherwise this'll just send to the CPU (which isnt as fast)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(agrs.decoder_path))

    # prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    # generate a caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy() # (1, max_seg_length) -> (max_seg_length)

    # convert word ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break

    sentence = ' '.join(sampled_caption)

    # print out the image and the generated caption
    print(sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-2-1000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-2-1000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')

    # model params (should be the same as in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in LSTM')
    args = parser.parse_args()
    main(args)