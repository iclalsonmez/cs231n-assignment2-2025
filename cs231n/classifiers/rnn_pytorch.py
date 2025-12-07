import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss on all parameters.
        """
        # Cut captions into two pieces
        captions_in = captions[:, :-1]   # (N, T)
        captions_out = captions[:, 1:]   # (N, T)
        mask = captions_out != self._null

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ############################################################################
        # Forward pass for the CaptioningRNN                                       #
        ############################################################################
        # 1. features -> initial hidden state (N, H)
        h0 = affine_forward(features, W_proj, b_proj)          # (N, H)

        # 2. word embedding (N, T, W)
        x_embed = word_embedding_forward(captions_in, W_embed) # (N, T, W)

        # 3. RNN / LSTM forward (N, T, H)
        if self.cell_type == "rnn":
            h = rnn_forward(x_embed, h0, Wx, Wh, b)            # (N, T, H)
        else:
            h = lstm_forward(x_embed, h0, Wx, Wh, b)           # (N, T, H)

        # 4. temporal affine to vocab scores (N, T, V)
        scores = temporal_affine_forward(h, W_vocab, b_vocab)  # (N, T, V)

        # 5. temporal softmax loss (scalar tensor)
        loss = temporal_softmax_loss(scores, captions_out, mask)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO: Implement test-time sampling for the model.                      #
        ###########################################################################
        # initial hidden state from image features
        h = affine_forward(features, W_proj, b_proj)  # (N, H)

        # initial cell state
        if self.cell_type == "lstm":
            c = torch.zeros_like(h)
        else:
            c = None  # unused

        # first input word is start
        prev_words = torch.full((N,), self._start, dtype=torch.long)

        for t in range(max_length):
            # 1. embed previous word: (N, W)
            x_t = W_embed[prev_words]  # (N, wordvec_dim)

            # 2. RNN/LSTM step
            if self.cell_type == "rnn":
                h = rnn_step_forward(x_t, h, Wx, Wh, b)  # (N, H)
            else:
                h, c = lstm_step_forward(x_t, h, c, Wx, Wh, b)  # (N, H), (N, H)

            # 3. compute scores over vocab: (N, V)
            scores = h @ W_vocab + b_vocab  # (N, V)

            # 4. pick highest
            next_words = scores.argmax(dim=1)  # (N,)
            captions[:, t] = next_words
            prev_words = next_words
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return captions