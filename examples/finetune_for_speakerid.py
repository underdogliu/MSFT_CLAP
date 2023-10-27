"""
This is a script for fine-tuning the CLAP model
for downstream task on speaker ID
"""
import os
import sys

sys.path.append(".")

from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from msclap import CLAP
from torch.utils.data import Dataset


class SpeakerDataset(Dataset):
    def __init__(self, train_list: str):
        self.audio_paths = []
        self.targets = []
        self.prompts = []

        with open(train_list, "r") as t:
            for line in t:
                _, spkid, wav_path = line.split()[:3]
                self.audio_paths.append(wav_path)
                self.targets.append(int(spkid))
                self.prompts.append(" ".join(line.split()[3:]))

    def __getitem__(self, index):
        file_path, one_hot_target, prompt = (
            self.audio_paths[index],
            self.targets[index],
            self.prompts[index],
        )
        return (file_path, one_hot_target, prompt)

    def __len__(self):
        return len(self.audio_paths)


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.contiguous()  # contiguous error
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )
        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super(ContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(
            torch.tensor([np.log(1 / temperature)]), requires_grad=True
        )

    def forward(self, h1, h2):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            h1 = SyncFunction.apply(h1)
            h2 = SyncFunction.apply(h2)
        device = h1.device
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        h1 = h1.squeeze()
        h2 = h2.squeeze()
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)

        logits = torch.einsum("nc,mc->nm", [h1, h2]) * temperature.to(device)
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    def acc(self, h1, h2):
        device = h1.device
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        h1 = h1.squeeze()
        h2 = h2.squeeze()
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum("nc,mc->nm", [h1, h2]) * temperature.to(device)
        N = logits.shape[0]
        y_pred = logits.max(dim=-1)[1]
        target = torch.arange(N, dtype=torch.long, device=device)
        train_acc = torch.sum(y_pred == target)
        acc = train_acc / N
        return acc


if __name__ == "__main__":
    data_dir = sys.argv[1]
    eval_data_dir = sys.argv[2]
    eval_out_dir = sys.argv[3]
    for modal in ["audio", "text"]:
        os.makedirs(eval_out_dir + "/{}".format(modal), exist_ok=True)

    # Load and initialize CLAP
    weights_path = "weights_path"
    clap_model = CLAP(version="2023", use_cuda=True)
    device = "cuda"
    clap_model = clap_model.to(device)

    # set clap model to be the training mode
    clap_model.clap.train()

    # initiate loss function
    cts_loss = ContrastiveLoss()

    # prepare the dataset
    spk_dataset = SpeakerDataset(data_dir + "/train_list.txt")

    # perform fine-tuning on the model
    for i in tqdm(range(len(spk_dataset))):
        x, one_hot_target, prompt = spk_dataset.__getitem__(i)
        audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
        text_embeddings = clap_model.get_text_embeddings([prompt])
        # similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        a_loss = cts_loss(audio_embeddings, text_embeddings)
        t_loss = cts_loss(text_embeddings, audio_embeddings)
        loss = (a_loss + t_loss) / 2
        loss.backward()

    # extract evaluation data
    eval_spk_dataset = SpeakerDataset(eval_data_dir + "/train_list.txt")

    for i in tqdm(range(len(eval_spk_dataset))):
        x, one_hot_target, prompt = spk_dataset.__getitem__(i)
        audio_embeddings = F.normalize(
            clap_model.get_audio_embeddings([x], resample=True)
        )
        text_embeddings = F.normalize(clap_model.get_text_embeddings([prompt]))

        # store embeddings
        utt_name = x.split("/")[-1].split(".")[0]
        audio_embeddings = audio_embeddings.detach().cpu().numpy()
        text_embeddings = text_embeddings.detach().cpu().numpy()
        np.save(
            eval_out_dir + "/audio/{}.npy".format(utt_name),
            audio_embeddings,
            allow_pickle=True,
        )
        np.save(
            eval_out_dir + "/text/{}.npy".format(utt_name),
            text_embeddings,
            allow_pickle=True,
        )
