import numpy as np

emb = np.load("/home/teaching/group46/CT_embeddings/train/embeddings/1BA001/000.npy")
print(emb.shape)  # (256, 64, 64)

# Convert to tensor
import torch
emb_tensor = torch.from_numpy(emb).unsqueeze(0)  # (1, 256, 64, 64)