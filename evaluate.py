import torch
import pickle

#
#
#
#
#

# Load vocab mappings
vocab_to_int = pickle.load(open('./tkn_vocab_to_int.pkl', 'rb'))
int_to_vocab = pickle.load(open('./tkn_int_to_vocab.pkl', 'rb'))


#
#
#
#
#

def topk(model, target_word="computer", top_k=5):
    if target_word not in vocab_to_int:
        print(f"[WARN] Word '{target_word}' not found in vocabulary.")
        return

    device = next(model.parameters()).device
    model.eval()

    idx = vocab_to_int[target_word]
    vec = model.emb.weight[idx].detach().to(device)

    with torch.no_grad():
        vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1)
        emb = torch.nn.functional.normalize(model.emb.weight.detach(), p=2, dim=1)
        sim = torch.matmul(emb, vec.squeeze())

        top_val, top_idx = torch.topk(sim, top_k + 1)  # +1 to skip the word itself
        print(f"\nTop {top_k} words similar to \"{target_word}\":")
        count = 0
        for i, idx in enumerate(top_idx):
            word = int_to_vocab.get(idx.item(), "<UNK>")
            if word == target_word:
                continue  # skip the word itself
            print(f"  {word}: {top_val[i].item():.4f}")
            count += 1
            if count == top_k:
                break