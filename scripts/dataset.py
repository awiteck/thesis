from torch.utils.data import Dataset
import torch


class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.

    """

    def __init__(
        self,
        data: torch.tensor,
        indices: list,
        enc_seq_len: int,
        target_seq_len: int,
    ) -> None:
        """
        Args:

            data: tensor, the entire train, validation or test data sequence
                        before any slicing. If univariate, data.size() will be
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence.
                     The sub-sequence is split into src, trg and trg_y later.

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """
        super().__init__()
        self.indices = indices
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]
        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]
        sequence = self.data[start_idx:end_idx]
        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            target_seq_len=self.target_seq_len,
        )

        return src, trg, trg_y

    def get_src_trg(
        self,
        sequence: torch.Tensor,
        enc_seq_len: int,
        target_seq_len: int,
    ):
        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence.

        Args:

            sequence: tensor, a nxk tensor where
                    n = encoder input length + target sequence length
                    k = number of input features

            enc_seq_len: int, the desired length of the input to the transformer encoder

            target_seq_len: int, the desired length of the target sequence (the
                            one against which the model output is compared)

        Return:

            src: tensor, 2D, used as input to the transformer model

            trg: tensor, 2D, used as input to the transformer model

            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss.

        """
        assert (
            len(sequence) == enc_seq_len + target_seq_len
        ), "Sequence length does not equal (input length + target length)"

        # encoder input
        src = sequence[:enc_seq_len]

        # decoder input. As per the paper, it must have the same dimension as the
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len - 1 : len(sequence) - 1]

        assert (
            len(trg) == target_seq_len
        ), "Length of trg does not match target sequence length"
        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]
        assert (
            len(trg_y) == target_seq_len
        ), "Length of trg_y does not match target sequence length"
        return (
            src,
            trg,
            trg_y
            # trg_y[:, 0].squeeze(-1),
        )  # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]


class BasicDataset(Dataset):
    """
    Dataset class used for basic linear model.
    """

    def __init__(
        self,
        data: torch.tensor,
        indices: list,
        enc_seq_len: int,
        target_seq_len: int,
    ) -> None:
        super().__init__()
        self.indices = indices
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]
        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]
        sequence = self.data[start_idx:end_idx]
        src, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            target_seq_len=self.target_seq_len,
        )

        return src, trg_y

    def get_src_trg(
        self,
        sequence: torch.Tensor,
        enc_seq_len: int,
        target_seq_len: int,
    ):
        assert (
            len(sequence) == enc_seq_len + target_seq_len
        ), "Sequence length does not equal (input length + target length)"

        # encoder input
        src = sequence[:enc_seq_len]

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]
        assert (
            len(trg_y) == target_seq_len
        ), "Length of trg_y does not match target sequence length"
        return (
            src,
            trg_y
            # trg_y[:, 0].squeeze(-1),
        )  # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]
