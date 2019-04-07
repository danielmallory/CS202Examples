import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        # load pretrained resnet-152 and replace the top fully connected layer
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    # this is the core of the computation; it takes the input images, runs them through the convolutions, then
    # reshapes the feature vectors (i.e. the convolved features created by passing the image tensors through the net)
    # into a completely flattened 1D tensor (i.e. a column vector) to run through the fully connected network
    # and passes them through the fully connected network with batch normalization.

    # the 'forward' method written here is an override from nn.Module, which is a generic class in pytorch.nn
    # that acts sort of like Java's "Object" class, but for neural network architectures. The forward function runs
    # forward propagation.
    def forward(self, images):
        # extract feature vectors from input images
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)

        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seg_length=20):
        # set the hyperparameters and build the layers
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seg_length

    def forward(self, features, captions, lengths):
        # decodes image feature vectors and generates captions
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        # generate captions for given image features using greedy search
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
