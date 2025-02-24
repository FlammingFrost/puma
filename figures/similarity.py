import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load .npy file
similarities = np.load("train/results/cosine_similarities.npy")

plt.figure(figsize=(10, 6))
sns.kdeplot(similarities, bw_adjust=0.5, color='g', fill=True, alpha=0.6)
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.title('Density of Cosine Similarities')
plt.show()

