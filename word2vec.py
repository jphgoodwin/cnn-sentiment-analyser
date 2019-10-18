import torch
import torch.nn as nn
import torch.nn.functional as fn
import pdb
import data_loader
import os

# Use gpu if available, otherwise use cpu.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SkipGram(nn.Module):
    # Class constructor.
    def __init__(self, v_size, d_size):
        # Call parent class constructor.
        super(SkipGram, self).__init__()

        # Store vocabulary and hidden layer sizes for use later.
        self.vocab_size = v_size
        self.hidden_size = d_size

        # Embedding layer that will contain the vector represenetations of the vocabulary.
        self.embeddings = nn.Embedding(v_size, d_size)

        # Linear layer that will transform embedding vectors back into one-hot-encoded vectors
        # with length equal to the size of the vocabulary.
        self.linear = nn.Linear(d_size, v_size)

        # LogSoftmax layer used in training to feed into the NLLLoss function.
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Softmax layer used during validation and testing to allow retrieval of word predictions.
        self.softmax = nn.Softmax(dim=1)

        # Negative Log Loss function used for backpropagation.
        self.loss = nn.NLLLoss()

    # Feedforward function returns embedding vectors from embedding matrix for one or more word indices.
    def forward(self, x):
        # Pass input through embedding layer to retrieve word vector(s).
        out = self.embeddings(x)
        return out

# Function trains embedding parameters using training_data provided, in batches of size bs, and
# for the specified number of epochs. The learning rate must also be specified, and you can
# optionally set the context window size (default 1), and whether or not to use padding. If
# validation_data is provided then this will be used to track training progress.
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

    # Create an optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            x_indxs = torch.tensor(x_indxs, device=device)
            y_indxs = torch.tensor(y_indxs, dtype=torch.long, device=device)
            
            # Zero gradients before running batch through model.
            optimizer.zero_grad()

            # Do a forward pass of x_indxs through embedding, linear, and logsoftmax layers
            # to generate y_preds values.
            y_preds = model.logsoftmax(model.linear(model(x_indxs)))

            # Calculate the negative log loss between y_preds and y_indxs.
            loss = model.loss(y_preds, y_indxs)

            # Do a backward pass using autograd to calculate gradient of loss with respect to
            # model parameters.
            loss.backward()

            # Update model parameters using calculated gradients.
            optimizer.step()

        # Empty GPU memory cache.
        torch.cuda.empty_cache()

        # If there is validation data, use it to test the performance of the network.
        if validation_data:
            # Create example tuples for each word paired with the words in its context window, and
            # concatenate into a single examples list.
            examples = []
            for wn in range(context, len(validation_data) - context):
                for cw in range(1, context+1):
                    examples.append((validation_data[wn], validation_data[wn-cw]))
                    examples.append((validation_data[wn], validation_data[wn+cw]))

            # Break the examples down into batches of size bs.
            batches = [examples[en:en+bs] for en in range(0, len(examples), bs)]

            # Run through model without calculating gradients, as we don't need them at this stage.
            with torch.no_grad():
                num_correct = 0
                # Iterate over batches.
                for batch in batches:
                    # Create lists of input and target output word indices.
                    x_indxs = []
                    y_indxs = []
                    for x, y in batch:
                        # Add input index to list.
                        x_indxs.append(x)
                        # Add output index to list.
                        y_indxs.append(y)

                    # Convert input index list to tensor.
                    x_indxs = torch.tensor(x_indxs, device=device)

                    # Do a foward pass through embedding, linear and softmax layers to generate y_preds.
                    # Round all values greater than 0.1 to 1 and the rest to 0.
                    y_preds = model.softmax(model.linear(model(x_indxs))).gt(0.1).type(torch.long)

                    # Iterate over examples in batch and count matches between predicted and actual words.
                    for n in range(0, len(batch)):
                        # Extract predicted and actual word indices.
                        p_indxs = torch.nonzero(y_preds[n])
                        a_indx = y_indxs[n]

                        # Determine whether a_indx is included in p_indxs and increment num_correct if so.
                        if ((p_indxs == a_indx).nonzero().nelement() > 0):
                            num_correct += 1

                # Clear GPU cache.
                torch.cuda.empty_cache()

                # Print results.
                print("Epoch {0}: {1} / {2}".format(i, num_correct, len(examples)))


# Function tests model on provided dataset and prints comparison of predicted words against
# actual words in string format if vocabulary provided, otherwise as indices.
def test(model, test_data, bs, context=1, padding=True, vocab=None, printWords=False):
    # Add padding equal to size of context window if required.
    if padding:
        for p in range(0, context):
            # Add padding to beginning. Padding is represented by "##" in index 1 of vocabulary.
            test_data.insert(0, 1)
            # Add padding to end.
            test_data.append(1)

    # Create example tuples for each word paired with the words in its context window, and
    # concatenate into a single examples list.
    examples = []
    for wn in range(context, len(test_data) - context):
        for cw in range(1, context+1):
            examples.append((test_data[wn], test_data[wn-cw]))
            examples.append((test_data[wn], test_data[wn+cw]))

    # Break the examples down into batches of size bs.
    batches = [examples[en:en+bs] for en in range(0, len(examples), bs)]

    # Run through model without calculating gradients as we don't need them when testing.
    with torch.no_grad():
        num_correct = 0
        # Iterate over batches.
        for batch in batches:
            # Create lists of input and target output word indices.
            x_indxs = []
            y_indxs = []
            for x, y in batch:
                # Add input index to list.
                x_indxs.append(x)
                # Add output index to list.
                y_indxs.append(y)

            # Convert input index list to tensor.
            x_indxs = torch.tensor(x_indxs, device=device)

            # Do a forward pass through embedding, linear and softmax layers to generate y_preds.
            # Round all values greater that 0.1 to 1 and the rest to 0.
            y_preds = model.softmax(model.linear(model(x_indxs))).gt(0.1).type(torch.long)

            # Iterate over examples in batch and count matches between prediced and actual words.
            for n in range(0, len(batch)):
                # Extract predicted and actual word indices.
                p_indxs = torch.nonzero(y_preds[n])
                a_indx = y_indxs[n]

                # Determine whether a_indx is included in p_indxs and increment num_correct if so.
                if ((p_indxs == a_indx).nonzero().nelement() > 0):
                    num_correct += 1

                if printWords and vocab:
                    # Map to word strings.
                    x_word = vocab[x_indxs[n]]
                    p_words = [vocab[w] for w in p_indxs]
                    a_word = vocab[a_indx]

                    # Print words
                    print("x: {0}, y_pred: {1}, y_act: {2}".format(x_word, p_words, a_word))
                elif printWords:
                    # Print indices.
                    print("x: {0}, y_pred: {1}, y_act: {2}".format(x_indx, p_indxs, a_indx))

        # Clear GPU cache.
        torch.cuda.empty_cache()

        # Print results.
        print("Test results: {0} / {1}".format(num_correct, len(examples)))


def saveModel(model, model_name):
    try:
        # See if model_name directory already exists.
        os.listdir("./models/" + model_name)
    except FileNotFoundError:
        # If not, make it.
        os.makedirs("./models/" + model_name)

    # Save the model parameters to a file within the model_name directory.
    torch.save(model.state_dict(), "./models/" + model_name + "/state_dict.pt")


def loadModel(model, model_name):
    try:
        # Load the saved model parameters into our model instance.
        model.load_state_dict(torch.load("./models/" + model_name + "/state_dict.pt"))
    except FileNotFoundError:
        # Alert user if files don't exist.
        print("Saved file not found.")

# Load IMDB dataset.
dl = data_loader.IMDBDataLoader("./data/aclImdb/imdb.vocab", "./data/aclImdb/train/", "./data/aclImdb/test/", 10, 10)

# Create SkipGram model instance.
sg = SkipGram(len(dl.vocab), 10)

loadModel(sg, "model_2")

# Move model onto GPU if available.
sg = sg.to(device)

# Extract training data as list of words, concatenating examples together.
tr_data = []
for ex in dl.ptrainex:
    tr_data.extend(ex[2])

# Reuse training examples for validation.
va_data = tr_data[:10000]

# Train model with training dataset.
train(model=sg, training_data=tr_data, epochs=5, bs=2048, lr=0.01, context=2, validation_data=va_data)

# saveModel(sg, "gcp_model_5")

# Extract test data similarly to training data.
te_data = []
for ex in dl.ptestex:
    te_data.extend(ex[2])

# Test model with test dataset.
test(model=sg, test_data=te_data, bs=2048, context=2, vocab=dl.vocab, printWords=False)
