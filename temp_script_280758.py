#!/usr/bin/env python
# coding: utf-8

# In[1]:


from llm_utils.activation_generator import ActivationGenerator
from data_utils.concept_dataset import SupervisedConceptDataset
import random
import numpy as np
import torch


# In[2]:


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# In[3]:


# path to data
data_path = "data/languages.json"

# name of model, must be supported by transformer lens
model_name = "gpt2-large"

# layers of model to inspect
layers = [0]

# Device to load data to, default is CPU and only factorization and generation occurs on GPU
data_device = 'cpu'

# Device to load model to for generation
model_device = device

# factorization mode, factorize residual or mlp layers
factorization_mode = 'mlp'


# In[4]:


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior

set_seed(42)


# In[5]:


act_generator = ActivationGenerator(model_name, model_device=model_device, data_device=data_device, mode=factorization_mode)


# In[6]:


dataset = SupervisedConceptDataset(data_path)


# In[7]:


activations, freq = act_generator.generate_multiple_layer_activations_and_freq(dataset, layers)


# In[8]:


# model parameters
rank = 100 # number of features
nmf_device = 'mps' # your GPU
fitting_device = device # your GPU
max_iterations = 20000 # has early stopping this is just the max
epoch_to_epoch_tol = 1e-6 # diff. in training objective that we tolerate (if smaller we end training)
# set above to negative if don't want early stopping
lr = 1e-3 # 1e-2 till 1e-4 works well (lower is just slower)


# In[9]:


from factorization.seminmf import NMFSemiNMF
# sparsity is percent of neurons to use in final features
nmf = NMFSemiNMF(rank, fitting_device=fitting_device, sparsity=0.01)

# patience is how many epochs to wait for loss to improve
# init can be svd and knn too, in terms of performance they are all the same
# we need to tranpose activations to match literature's (dimensio, num_samples)
# we take activations[0] since its a list of activations (index for every layer you used when generating)
nmf.fit(activations[0].T, max_iterations, patience=500)


# #### Utils for interpreting features

# In[10]:


def get_top_activating_indices(W, concept_idx, num_samples=10, minimal_activation=0):
    activations = []
    non_zero_indices = []

    sample_importance = W[:, concept_idx]
    # Get indices of the top samples (highest activation values)
    top_indices = np.argsort(sample_importance)[-num_samples:]
    for i in top_indices:
        act = sample_importance[i]
        if act <= 0:
            continue
        activations.append(act)
        non_zero_indices.append(i)

    return non_zero_indices, activations

def print_logit_diff(model, logits_before, logits_after, top_k=10):
    """
    Print the tokens with the largest positive and negative logit changes
    after some intervention.

    Args:
      model         : object with `to_str_tokens(torch.LongTensor)->List[str]`
      logits_before : torch.Tensor, shape (1, seq_len, vocab_size)
      logits_after  : torch.Tensor, shape (1, seq_len, vocab_size)
      top_k         : int, how many tokens to show in each category
    """
    # compute delta for the last position
    delta = logits_after[0, -1, :] - logits_before[0, -1, :]

    # top positive changes
    pos_vals, pos_idx = torch.topk(delta, k=top_k, largest=True)
    # top negative changes
    neg_vals, neg_idx = torch.topk(delta, k=top_k, largest=False)

    print(f"Top {top_k} ↑ logit changes:")
    for token_id, change in zip(pos_idx.tolist(), pos_vals.tolist()):
        ids = torch.tensor([token_id], dtype=torch.long, device=delta.device)
        token_str = model.to_str_tokens(ids)[0]
        print(f"  {token_str:>12}   {change:+.4f}")

    print(f"\nTop {top_k} ↓ logit changes:")
    for token_id, change in zip(neg_idx.tolist(), neg_vals.tolist()):
        ids = torch.tensor([token_id], dtype=torch.long, device=delta.device)
        token_str = model.to_str_tokens(ids)[0]
        print(f"  {token_str:>12}   {change:+.4f}")


def get_logit_diff(model, logits_before, logits_after, top_k=20, magnitude=False):
    """
    Return a list of token strings and their logit changes.

    Args:
      model         : object with `to_str_tokens(torch.LongTensor) -> List[str]`
      logits_before : torch.Tensor, shape (1, seq_len, vocab_size)
      logits_after  : torch.Tensor, shape (1, seq_len, vocab_size)
      top_k         : int, how many tokens to return
      magnitude     : bool, if True rank by abs(delta), else by signed delta

    Returns:
      List[str], e.g. ["Token: 'hello', Score: 2.3456", …]
    """
    delta = logits_after[0, -1, :] - logits_before[0, -1, :]
    if magnitude:
        scores, idx = torch.topk(delta.abs(), k=top_k)
    else:
        scores, idx = torch.topk(delta, k=top_k)

    results = []
    for token_id, score in zip(idx.tolist(), scores.tolist()):
        ids_tensor = torch.tensor([token_id], dtype=torch.long, device=logits_before.device)
        token_strs = model.to_str_tokens(ids_tensor)
        results.append(f"Token: {token_strs[0]}, Score: {score:.4f}")
    return results


# #### Helper function to extract the tokens and labels for interpreting features

# In[11]:


from llm_utils.activation_generator import extract_token_ids_sample_ids_and_labels

tokens, sample_ids, labels = extract_token_ids_sample_ids_and_labels(dataset, act_generator)


# #### Helper function to get token contexts
# So we see the contexts and not just the token whose activation was activated

# In[12]:


def generate_token_contexts(tokens, sample_ids, act_generator):
    # Define how many tokens before and after to include in the context
    context_window = 15

    token_ds = []
    for i in range(len(tokens)):
        current_sample_id = sample_ids[i]
        # Convert the current token to its string representation
        token_str = act_generator.model.to_str_tokens([tokens[i]])[0][0]

        # Determine the start and end indices for the context window
        start = max(0, i - context_window)
        end = min(len(tokens), i + context_window + 1)

        # Get the string representation for each token in the context
        context_tokens = [
            act_generator.model.to_str_tokens([tokens[j]])[0][0] for j in range(start, end) if sample_ids[j] == current_sample_id
        ]

        # Join the context tokens into a single string
        context_str = "".join(context_tokens)

        # Append the (token, context) tuple to the list
        token_ds.append((token_str, context_str))

    return token_ds

token_ds = generate_token_contexts(tokens, sample_ids, act_generator)


# #### Prints for the first 50 features their activating token, activation strengh, context

# In[13]:


for k in range(50):
    ti, ta = get_top_activating_indices(nmf.G_.cpu().detach(), k, 40)
    top_activations = [{'token': token_ds[i][0], 'activation': a, 'context': token_ds[i][1]} for i, a in zip(ti, ta)]
    print(f"###########{k}#############\n")
    for idx, i in enumerate(ti):
        print(f"{token_ds[i][0]}\t\t{ta[idx]}\t\t{token_ds[i][1]}")


# ### Interpreting Through Intervention

# In[14]:


from intervention.intervener import Intervener

intervener = Intervener(act_generator.model)


# #### Token Change (shows tokens whose logits changed the most due to intervention)

# In[15]:


base_prompt = "I think that"


# In[16]:


with torch.no_grad():
    base_logits = act_generator.model(act_generator.model.to_tokens(base_prompt))


# In[17]:


alpha = 20
intervention_layer = 0

for i in range(50):

    print(f"#####################################{i}###########################")
    intervened_logits = intervener.intervene(
                    base_prompt,
                    [nmf.F_.T[i].to(device)], 
                    layers=[intervention_layer], 
                    alpha=alpha, 

    )
    print_logit_diff(intervener._model, base_logits, intervened_logits)



# In[ ]:




