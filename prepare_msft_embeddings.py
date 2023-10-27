"""
Prepare audio and text embedding and run the benchmarks
"""
import os
import sys

import numpy as np
from src import CLAP


if __name__ == "__main__":
    src_dir = sys.argv[1]
    exp_dir = sys.argv[2]
    os.makedirs(exp_dir + "/embeddings/audio", exist_ok=True)
    os.makedirs(exp_dir + "/embeddings/text", exist_ok=True)

    clap_model = CLAP("<PATH TO WEIGHTS>", version="2023", use_cuda=False)

    # load data
    ext_data = {}
    with open(src_dir + "/train_list.txt", "r") as w:
        for line in w:
            line_sp = line.split()
            utt = line_sp[0]
            wav_path = line_sp[2]
            prompt = " ".join(line_sp[3:])

            ext_data[utt] = [wav_path, prompt]

    # extract embeddings
    for utt in ext_data.keys():
        wav_path, prompt = ext_data[utt]

        audio_embed = (
            clap_model.get_audio_embeddings([wav_path], resample=True)
            .squeeze()
            .cpu()
            .numpy()
        )
        audio_embed_path = exp_dir + "/embeddings/audio/{}.npy".format(utt)
        np.save(audio_embed_path, audio_embed, allow_pickle=True)

        text_embed = clap_model.get_text_embeddings(prompt).squeeze().cpu().numpy()
        text_embed_path = exp_dir + "/embeddings/text/{}.npy".format(utt)
        np.save(text_embed_path, text_embed, allow_pickle=True)
