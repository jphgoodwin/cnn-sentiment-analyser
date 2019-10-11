import torch
import torch.nn as nn
import torch.nn.functional as fn
import pdb
import data_loader

class SkipGram(nn.Module):
    # Class constructor.
    def __init__(self, v_size, d_size):
        # Call parent class constructor.
        super(SkipGram, self).__init__()

        # Store vocabulary and hidden layer sizes for use later.
        self.vocab_size = v_size
        self.hidden_size = d_size

        self.embeddings = nn.Embedding(v_size, d_size)
        self.linear = nn.Linear(d_size, v_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.NLLLoss()

    def forward(self, x):
        # Pass input through layers sequentially.
        out1 = self.embeddings(x)
        out2 = self.linear(out1)
        return out2

def train(model, training_data, epochs, bs, lr, context=1, padding=True, validation_data=None):
    # Add padding equal to size of context window if required.
    if padding:
        for p in range(0, context):
            # Add padding to beginning. Padding is represented by "##" in index 1 of vocabulary.
            training_data.insert(0, 1)
            # Add padding to end.
            training_data.append(1)
            
            # Add padding to validation data if present.
            if validation_data:
                validation_data.insert(0, 1)
                validation_data.append(1)

    # Break example range down into batches, accounting for the required context window
    # before the first and after the last examples. If no padding has been added, this will
    # mean a context window worth of examples at each end will be excluded.

    # Create example tuples for each word paired with the words in its context window, and
    # concatenate into a single examples list. The required context window is accounted for
    # in the range, so if no padding has been added some words at the beginning and end
    # will be excluded from training.
    examples = []
    for wn in range(context, len(training_data) - context):
        for cw in range(1, context+1):
            examples.append((training_data[wn], training_data[wn-cw]))
            examples.append((training_data[wn], training_data[wn+cw]))

    # Break the examples down into batches of size bs.
    batches = [examples[en:en+bs] for en in range(0, len(examples), bs)]

    # Train for the specified number of epochs.
    for i in range(1, epochs+1):

        # Iterate over the batches.
        for batch in batches:
            # Create lists of input and target output word indexes.
            x_indxs = []
            y_indxs = []
            for x, y in batch:
                # Add input index to list.
                x_indxs.append(x)
                # Add output index to list.
                y_indxs.append(y)

            # Convert index lists to tensor.
            x_indxs = torch.tensor(x_indxs)
            y_indxs = torch.tensor(y_indxs, dtype=torch.long)

            # Do a forward pass of x_indxs through the network to generate y_preds values.
            y_preds = model.logsoftmax(model(x_indxs))

            # Zero gradients before running the backward pass.
            model.zero_grad()

            # Calculate the negative log loss between y_preds and y_indxs.
            loss = model.loss(y_preds, y_indxs)

            # Do a backward pass using autograd to calculate gradient of loss with respect to
            # model parameters.
            loss.backward()

            # Update model parameters using calculated gradients.
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad

        # If there is validation data, use it to test the performance of the network.
        if validation_data:
            # Create example tuples for each word paired with the words in its context window, and
            # concatenate into a single examples list.
            examples = []
            for wn in range(context, len(validation_data) - context):
                for cw in range(1, context+1):
                    examples.append((validation_data[wn], validation_data[wn-cw]))
                    examples.append((validation_data[wn], validation_data[wn+cw]))

            test_results = []
            # Iterate over validation data.
            for x, y in examples:
                # Set x_indx to input word index.
                x_indx = torch.tensor([x], dtype=torch.long)

                # Create output one-hot-encoded word vector from output word index.
                y_vec = torch.zeros(model.vocab_size, dtype=torch.long)
                y_vec[y] = 1

                # Add tuple of predicted and actual output vectors to test_results.
                test_results.append((model.softmax(model(x_indx)), y_vec))

            num_correct = 0
            # Iterate over test_results and compare predicted to actual output, counting the
            # number of predicted vectors that contain the actual vector for each example.
            for y_pred, y_act in test_results:
                # Round all values greater than or equal to 0.1 to 1 and the rest to 0.
                y_pred.gt_(0.1).type(torch.LongTensor)

                # Extract indices.
                p_indxs = torch.nonzero(y_pred)
                a_indx = torch.nonzero(y_act)

                # Determine whether y_act is included in y_pred and increment num_correct if so.
                if ((p_indxs == a_indx).nonzero().nelement() > 0):
                    num_correct += 1

            # Print results.
            print("Epoch {0}: {1} / {2}".format(i, num_correct, len(test_results)))


dl = data_loader.IMDBDataLoader("./data/aclImdb/imdb.vocab", "./data/aclImdb/train/", "./data/aclImdb/test/", 3, 3)

# Extract training data as list of words, concatenating examples together.
tr_data = []
for ex in dl.ptrainex:
    tr_data.extend(ex[2])
va_data = tr_data[:]
te_data = []
for ex in dl.ptestex:
    te_data.extend(ex[2])

sg = SkipGram(len(dl.vocab), 10)

train(model=sg, training_data=tr_data, epochs=50, bs=16, lr=0.005, context=1, validation_data=va_data)

# x_in = torch.tensor(0)
# x_out = sg(x_in)

# print(sg.embeddings.weight)
# print(x_out)
