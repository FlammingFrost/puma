import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load .npy file
similarities = np.load("train/results/cosine_similarities.npy")
mlp_similarities = np.load("train/results/mlp_cosine_similarities.npy")
small_lm_similarities = np.load("train/results/small_lm_cosine_similarities.npy")
small_mlp_similarities = np.load("train/results/small_mlp_cosine_similarities.npy")
small_lora_similarities = np.load("train/results/lora_cosine_similarities_small.npy")





# plot

plt.figure(figsize=(10, 6))
sns.kdeplot(similarities, bw_adjust=0.5, color='g', fill=True, alpha=0.6, label='Pre-trained')
sns.kdeplot(mlp_similarities, bw_adjust=0.5, color='r', fill=True, alpha=0.6, label='MLP-transformed')
sns.kdeplot(small_lm_similarities, bw_adjust=0.5, color='b', fill=True, alpha=0.6, label='Small LM')
sns.kdeplot(small_mlp_similarities, bw_adjust=0.5, color='y', fill=True, alpha=0.6, label='Small MLP')
sns.kdeplot(small_lora_similarities, bw_adjust=0.5, color='m', fill=True, alpha=0.6, label='Small LoRA')
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

