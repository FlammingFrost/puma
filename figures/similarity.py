import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load .npy file
similarities = np.load("train/results/cosine_similarities.npy")
mlp_similarities = np.load("train/results/mlp_cosine_similarities.npy")
lora_similarities = np.load("train/results/lora_cosine_similarities.npy")

# plot

plt.figure(figsize=(10, 6))
sns.kdeplot(similarities, bw_adjust=0.5, color='g', fill=True, alpha=0.6, label='Pre-trained')
sns.kdeplot(mlp_similarities, bw_adjust=0.5, color='r', fill=True, alpha=0.6, label='MLP-transformed')
# sns.kdeplot(lora_similarities, bw_adjust=0.5, color='b', fill=True, alpha=0.6)
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.title('Density of Cosine Similarities')
# add legend
plt.legend()
# plt.show()
plt.savefig("figures/similarity.png")

# summarize lora similarities distribution
# print("Lora similarities:")
# print("Mean:", np.mean(lora_similarities))
# print("Std:", np.std(lora_similarities))
# print("Min:", np.min(lora_similarities))
# print("Max:", np.max(lora_similarities))

