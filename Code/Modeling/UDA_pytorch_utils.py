"""
Helper code for Carnegie Mellon University's Unstructured Data Analytics course
Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import collate_tensors


def UDA_pytorch_classifier_fit(model, optimizer, loss,
                               proper_train_dataset, val_dataset,
                               num_epochs, batch_size, device=None,
                               sequence=False):
    """
    Trains a neural net classifier `model` using an `optimizer` such as Adam or
    stochastic gradient descent. We specifically minimize the given `loss`
    (e.g., categorical or binary cross entropy) using the data given by
    `proper_train_dataset` using the number of epochs given by `num_epochs` and
    a batch size given by `batch_size`.

    Accuracies on the (proper) training data (`proper_train_dataset`) and
    validation data (`val_dataset`) are computed at the end of each epoch;
    `val_dataset` can be set to None if you don't want to use a validation set.
    The function outputs the training and validation accuracies.

    You can manually set which device (CPU or GPU) to use with the optional
    `device` argument (e.g., setting `device=torch.device('cpu')` or
    `device=torch.device('cuda')`). By default, the code tries to use a GPU if
    it is available.

    Lastly, the boolean argument `sequence` says whether we are looking at time
    series data (set this True for working with recurrent neural nets).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not sequence:
        # PyTorch uses DataLoader to load data in batches
        proper_train_loader = \
            torch.utils.data.DataLoader(dataset=proper_train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    else:
        proper_train_loader = \
                UDA_get_batches_sequence(proper_train_dataset,
                                         batch_size,
                                         shuffle=True,
                                         device=device)
        val_loader = \
                UDA_get_batches_sequence(val_dataset,
                                         batch_size,
                                         shuffle=False,
                                         device=device)

    proper_train_size = len(proper_train_dataset)
    val_size = len(val_dataset)

    train_accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)

    for epoch_idx in range(num_epochs):
        # go through training data
        num_training_examples_so_far = 0
        for batch_idx, (batch_features, batch_labels) \
                in enumerate(proper_train_loader):
            # make sure the data are stored on the right device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # make predictions for current batch and compute loss
            batch_outputs = model(batch_features)
            batch_loss = loss(batch_outputs, batch_labels)

            # update model parameters
            optimizer.zero_grad()  # reset which direction optimizer is going
            batch_loss.backward()  # compute new direction optimizer should go
            optimizer.step()  # move the optimizer

            # draw fancy progress bar
            num_training_examples_so_far += batch_features.shape[0]
            sys.stdout.write('\r')
            sys.stdout.write("Epoch %d [%-50s] %d/%d"
                             % (epoch_idx + 1,
                                '=' * int(num_training_examples_so_far
                                          / proper_train_size * 50),
                                num_training_examples_so_far,
                                proper_train_size))
            sys.stdout.flush()

        # draw fancy progress bar at 100%
        sys.stdout.write('\r')
        sys.stdout.write("Epoch %d [%-50s] %d/%d"
                         % (epoch_idx + 1,
                            '=' * 50,
                            num_training_examples_so_far, proper_train_size))
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

        # compute proper training and validation set raw accuracies
        train_accuracy = \
                UDA_pytorch_classifier_evaluate(model,
                                                proper_train_dataset,
                                                device=device,
                                                batch_size=batch_size,
                                                sequence=sequence)
        print('  Train accuracy: %.4f' % train_accuracy, flush=True)
        train_accuracies[epoch_idx] = train_accuracy

        val_accuracy = \
                UDA_pytorch_classifier_evaluate(model,
                                                val_dataset,
                                                device=device,
                                                batch_size=batch_size,
                                                sequence=sequence)
        print('  Validation accuracy: %.4f' % val_accuracy, flush=True)
        val_accuracies[epoch_idx] = val_accuracy

    return train_accuracies, val_accuracies


def UDA_pytorch_model_transform(model, inputs, device=None, batch_size=128,
                                sequence=False):
    """
    Given a neural net `model`, evaluate the model given `inputs`, which should
    *not* be already batched. This helper function automatically batches the
    data, feeds each batch through the neural net, and then unbatches the
    outputs. The outputs are stored as a PyTorch tensor.

    You can manually set which device (CPU or GPU) to use with the optional
    `device` argument (e.g., setting `device=torch.device('cpu')` or
    `device=torch.device('cuda')`). By default, the code tries to use a GPU if
    it is available.

    You can also manually set `batch_size`; this is less critical than in
    training since we are, at this point, just evaluating the model without
    updating its parameters.

    Lastly, the boolean argument `sequence` says whether we are looking at time
    series data (set this True for working with recurrent neural nets).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # batch the inputs
    if not sequence:
        feature_loader = torch.utils.data.DataLoader(dataset=inputs,
                                                     batch_size=batch_size,
                                                     shuffle=False)
    else:
        feature_loader = \
                UDA_get_batches_from_encoded_text(inputs,
                                                  None,
                                                  batch_size,
                                                  shuffle=False,
                                                  device=device)

    outputs = []
    with torch.no_grad():
        idx = 0
        for batch_features in feature_loader:
            batch_features = batch_features.to(device)
            batch_outputs = model(batch_features)
            outputs.append(batch_outputs)

    return torch.cat(outputs, 0)


def UDA_pytorch_classifier_predict(model, inputs, device=None, batch_size=128,
                                   sequence=False):
    """
    Given a neural net classifier `model`, predict labels for the given
    `inputs`, which should *not* be already batched. This helper function
    automatically batches the data, feeds each batch through the neural net,
    and then computes predicted labels by looking at the argmax (for when
    categorical cross entropy is used) or applying sigmoid and thresholding at
    probability 1/2 (for when binary cross entropy is used). The output
    predicted labels are stored as a PyTorch tensor.

    You can manually set which device (CPU or GPU) to use with the optional
    `device` argument (e.g., setting `device=torch.device('cpu')` or
    `device=torch.device('cuda')`). By default, the code tries to use a GPU if
    it is available.

    You can also manually set `batch_size`; this is less critical than in
    training since we are, at this point, just evaluating the model without
    updating its parameters.

    Lastly, the boolean argument `sequence` says whether we are looking at time
    series data (set this True for working with recurrent neural nets).
    """
    outputs = UDA_pytorch_model_transform(model,
                                          inputs,
                                          device=device,
                                          batch_size=batch_size,
                                          sequence=sequence)

    with torch.no_grad():
        if outputs.shape[1] == 1:
            return (1*(torch.sigmoid(outputs) >= 0.5)).view(-1)
        else:
            return outputs.argmax(axis=1).view(-1)


def UDA_pytorch_classifier_evaluate(model, dataset, device=None,
                                    batch_size=128, sequence=False):
    """
    Evaluate the raw accuracy of a neural net classifier `model` for a
    `dataset`, which should be a list of pairs of the format (input, label).
    
    You can manually set which device (CPU or GPU) to use with the optional
    `device` argument (e.g., setting `device=torch.device('cpu')` or
    `device=torch.device('cuda')`). By default, the code tries to use a GPU if
    it is available.
    
    You can also manually set `batch_size`; this is less critical than in
    training since we are, at this point, just evaluating the model without
    updating its parameters.
    
    Lastly, the boolean argument `sequence` says whether we are looking at time
    series data (set this True for working with recurrent neural nets).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not sequence:
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    else:
        loader = UDA_get_batches_sequence(dataset,
                                          batch_size,
                                          shuffle=False,
                                          device=device)

    with torch.no_grad():
        num_correct = 0.
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_outputs = model(batch_features)
            if batch_outputs.shape[1] == 1:
                batch_predicted_labels = \
                        1*(torch.sigmoid(batch_outputs) >= 0.5)
            else:
                batch_predicted_labels = batch_outputs.argmax(axis=1)
            if type(batch_labels) == np.ndarray:
                batch_predicted_labels = \
                        batch_predicted_labels.view(-1).cpu().numpy()
                num_correct += (batch_predicted_labels == batch_labels).sum()
            else:
                num_correct += \
                        (batch_predicted_labels.view(-1)
                         == batch_labels.to(device).view(-1)).sum().item()

    return num_correct / len(dataset)


def UDA_plot_train_val_accuracy_vs_epoch(train_accuracies, val_accuracies):
    """
    Helper function for plotting (proper) training and validation accuracies
    across epochs; `train_accuracies` and `val_accuracies` should be the same
    length, which should equal the number of epochs.
    """
    ax = plt.figure().gca()
    num_epochs = len(train_accuracies)
    plt.plot(np.arange(1, num_epochs + 1), train_accuracies, '-o',
             label='Training')
    plt.plot(np.arange(1, num_epochs + 1), val_accuracies, '-+',
             label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def UDA_compute_accuracy(labels1, labels2):
    """
    Computes the raw accuracy of two label sequences `labels1` and `labels2`
    agreeing. This helper function coerces both label sequences to be on the
    CPU, flattened, and stored as 1D NumPy arrays before computing the average
    agreement.
    """
    if type(labels1) == torch.Tensor:
        labels1 = labels1.detach().view(-1).cpu().numpy()
    elif type(labels1) != np.ndarray:
        labels1 = np.array(labels1).flatten()
    else:
        labels1 = labels1.flatten()

    if type(labels2) == torch.Tensor:
        labels2 = labels2.detach().view(-1).cpu().numpy()
    elif type(labels2) != np.ndarray:
        labels2 = np.array(labels2).flatten()
    else:
        labels2 = labels2.flatten()

    return np.mean(labels1 == labels2)


class UDA_LSTMforSequential(nn.Module):
    """
    This helper class allows for an LSTM to be used with nn.Sequential().
    """
    def __init__(self, input_size, hidden_size, return_sequences=False):
        super().__init__()
        self.return_sequences = return_sequences
        self.model = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             batch_first=True)  # axis 0 indexes data in batch

    def forward(self, x):
        # x should be of shape (batch size, sequence length, feature dimension)
        outputs, _ = self.model(x)
        if self.return_sequences:
            return outputs
        else:
            return outputs[:, -1, :]  # take last time step's output


def UDA_get_batches_sequence(dataset, batch_size, shuffle=True, device=None):
    """
    Helper function that does the same thing as
    `UDA_get_batches_from_encoded_text()` except that the input dataset is a
    list of pairs of the format (encoded text, label). This function
    basically converts the input format to be what is expected by
    `UDA_get_batches_from_encoded_text()` and then runs that function. See
    the documentation for that function to understand what the arguments are.
    """
    text_encoded = []
    labels = []
    for text, label in dataset:
        text_encoded.append(text)
        labels.append(label)
    return UDA_get_batches_from_encoded_text(text_encoded, labels,
                                             batch_size, shuffle, device)


def UDA_get_batches_from_encoded_text(text_encoded, labels, batch_size,
                                      shuffle=True, device=None):
    """
    Batches sequence data, where sequences within the same batch could have
    unequal lengths, so padding is needed to get their lengths to be the same
    for feeding to the neural net. The input text `text_encoded` should already
    be encoded so that each text sequence consists of word indices to represent
    indices into a vocabulary. The i-th element of `text_encoded` should have a
    label given by the i-th entry in `labels` (which will be converted to a
    PyTorch tensor). The batch size is specified by `batch_size`.

    If `shuffle` is set to True, a bucket sampling strategy is used that reduces
    how much padding is needed in different batches while injecting some
    randomness.

    You can manually set which device (CPU or GPU) to use with the optional
    `device` argument (e.g., setting `device=torch.device('cpu')` or
    `device=torch.device('cuda')`). By default, the code tries to use a GPU if
    it is available.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if shuffle:
        # use bucket sampling strategy to reduce the amount of padding needed
        sampler = torch.utils.data.sampler.SequentialSampler(text_encoded)
        loader = BucketBatchSampler(
            sampler, batch_size=batch_size, drop_last=False,
            sort_key=lambda i: text_encoded[i].shape[0])
    else:
        indices = list(range(len(text_encoded)))
        loader = torch.utils.data.DataLoader(dataset=indices,
                                             batch_size=batch_size,
                                             shuffle=False)

    if labels is None:
        batches = [collate_tensors([text_encoded[i] for i in batch],
                                   stack_tensors=stack_and_pad_tensors
                                  ).tensor.to(device)
                   for batch in loader]
    else:
        batches = [(collate_tensors([text_encoded[i] for i in batch],
                                    stack_tensors=stack_and_pad_tensors
                                   ).tensor.to(device),
                    torch.tensor([labels[i] for i in batch],
                                 dtype=torch.long).to(device).view(-1))
                   for batch in loader]
    return batches
