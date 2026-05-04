import torch #for tensors
import torch.nn.functional as F #to normalize attention
import matplotlib.pyplot as plt
import seaborn as sns

sentence = ["I", "love", "deep", "learning"] #input sequence
attention_weights = torch.tensor([0.1, 0.3, 0.4, 0.2]) #initial (raw) importance->attention weights

attention_weights = F.softmax(attention_weights, dim=0).detach().numpy() #ranking same,just raw values into probability 

sns.heatmap([attention_weights], annot=True, xticklabels=sentence, cmap="Blues")
plt.title("Attention Heatmap")
plt.show()

