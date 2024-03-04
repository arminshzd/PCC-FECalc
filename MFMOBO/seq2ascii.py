from typing import Any, AnyStr
import pickle
import torch
from tensordict.tensordict import TensorDict

class Seq2Ascii:
    """
    Class to convert list of sequences to tensor of properties
    """

    def __init__(self, psi_dict_dir: AnyStr) -> None:
        with open(psi_dict_dir, 'rb') as f:
            self.psi_dict = pickle.load(f)
        self.str2int = None
        self.int2tensor = None
        self.AAmap = {aa: i for i, aa in enumerate(self.psi_dict.keys())}

    def fit(self, seqs: list) -> None:
        """
        Fit the encoder and decoder to the sequences

        Args:
            seqs (list): list of sequences
        """
        self.str2int = {seq: i for i, seq in enumerate(seqs)}
        self.int2str = {val: seq for seq, val in self.str2int.items()}
        self.int2tensor = {i: self.encode(seq) for i, seq in enumerate(seqs)}
    
    def encode_to_int(self, seqs: list) -> int:
        return torch.tensor([self.str2int[i] for i in seqs])

    def encode(self, seqs: str) -> torch.Tensor:
        """
        Wrapper for encoding a list of or a single sequence

        Args:
            seqs (str or list): single of a list of sequences

        Returns:
            torch.Tensor: Tensor of integers
        """
        if isinstance(seqs, list):
            return self.encode_list(seqs)
        else:
            return self.encode_seq(seqs)
    
    def encode_seq(self, seq: str) -> torch.Tensor:
        tensor = [torch.Tensor([self.AAmap[i]]).long() for i in seq]
        return torch.stack(tensor, dim=1).view(1, -1)

    def encode_list(self, seq_list: list) -> torch.Tensor:
        """
        Encode list of sequences to tensor of properties

        Args:
            seq_list (list): list of sequences

        Returns:
            torch.Tensor: Nx24xL tensor of properties
        """
        tensor_list = []
        for seq in seq_list:
            tensor_list.append(self.encode_seq(seq))
        return torch.concat(tensor_list, dim=0)

    def translate_to_ord(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Translate ints to vectors

        Args:
            X (torch.Tensor): Nx1 tensor of ids

        Returns:
            torch.Tensor: Nx5 tensor of ords
        """
        data = []
        for i in range(ids.size(0)):
            batch = []
            for n in range(ids.size(1)):
                batch.append(self.int2tensor[ids[i, n].item()].squeeze())
            data.append(torch.stack(batch, dim=0))
        return torch.stack(data, dim=0)
    
    def decode(self, X: torch.Tensor) -> str:
        if len(X) > 1:
            return [self._decode(i) for i in X.squeeze()]
        else:
            return self._decode(X)
        
    def _decode(self, X: torch.Tensor) -> str:
        """
        Translate property tensor to list of strings

        Args:
            X (torch.Tensor): Nx24xL tensor of properties

        Returns:
            str: decoded sequence
        """
        return self.int2str[X.item()]

    def get_psi(self, seqs: list) -> torch.Tensor:
        """
        Generate property tensor from list of encoded inputs

        Args:
            seqs (list): (N, ) list of encoded sequences

        Returns:
            torch.Tensor: Nx24xL tensor of properties
        """
        tensor_list = []
        for seq in seqs:
            tensor_list.append(self.seq_to_psi(self.decoder[seq.item()]))
        return torch.stack(tensor_list, dim=0)
