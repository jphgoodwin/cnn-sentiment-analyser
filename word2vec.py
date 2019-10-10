import torch
import torch.nn as nn
import torch.nn.functional as fn

class SkipGram(nn.Module):
    # Class constructor.
    def __init__(self, v_size, d_size):
        # Call parent class constructor.
        super(SkipGram, self).__init__()

        self.embeddings = nn.Embedding(v_size, d_size)
        self.linear = nn.Linear(d_size, v_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Pass input through layers sequentially.
        out1 = self.embeddings(x)
        out2 = self.linear(out1)
        out3 = self.softmax(out2)
        return out3

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


sg = SkipGram(10, 5)

x_in = torch.tensor(0)
x_out = sg(x_in)

print(sg.embeddings.weight)
print(x_out)
