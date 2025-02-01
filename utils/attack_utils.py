from transformers import AutoTokenizer,AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from sentence_transformers import SentenceTransformer
import torch
import logging
from tqdm import tqdm
import time
import random

logger = logging.getLogger(__name__)

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids

def evaluate_acc_yk_batch(model, c_model, get_emb, va_dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type,
                 data_collator, device='cuda'):
    model.eval()
    c_model.eval()
    p_sent = {'input_ids': adv_passage_ids.to(device),
              'attention_mask': adv_passage_attention.to(device),
              'token_type_ids': adv_passage_token_type.to(device)}
    p_emb = get_emb(c_model, p_sent)
    sim_all=[]

    for idx, (data) in tqdm(enumerate(va_dataloader)):
        sim = torch.mm(data, p_emb.T).squeeze()
        sim_all.append(sim.sum().cpu().item())

    return sum(sim_all) / len(sim_all)



def hotflip_candidate(args, grad, embeddings,):
    token_to_flip = random.randrange(args.num_adv_passage_tokens)
    candidates = hotflip_attack(grad[token_to_flip],
                                      embeddings.weight,
                                      increase_loss=True,
                                      num_candidates=args.num_cand,
                                      filter=None)
    return token_to_flip , candidates

def hotflip_candidate_score(args, it_, candidates, pbar, train_iter, data_collator, get_emb, model, c_model,
                            adv_passage_ids, adv_passage_attention, adv_passage_token_type,token_to_flip, device):
    current_score = 0
    candidate_scores = torch.zeros(args.num_cand, device=device)
    current_acc_rate = 0
    candidate_acc_rates = torch.zeros(args.num_cand, device=device)

    for step in pbar:
        try:
            it_centrid_embedding = next(train_iter)
        except:
            print('Insufficient data!')
            break

        p_sent = {'input_ids': adv_passage_ids,
                  'attention_mask': adv_passage_attention,
                  'token_type_ids': adv_passage_token_type}
        p_emb = get_emb(c_model, p_sent)
        # Compute loss
        sim = torch.mm(it_centrid_embedding, p_emb.T)  # [b x k]
        loss = sim.mean()
        temp_score = loss.sum().cpu().item()

        current_score += temp_score

        start_time = time.time()
        batch_size = len(candidates)
        temp_adv_passage = adv_passage_ids.clone().repeat(batch_size, 1)  # [num_candidates, 50]

        temp_adv_passage[:, token_to_flip] = candidates

        p_sent = {
            'input_ids': temp_adv_passage,  # [num_candidates, 50]
            'attention_mask': adv_passage_attention.repeat(batch_size, 1),
            'token_type_ids': adv_passage_token_type.repeat(batch_size, 1)
        }

        p_emb = get_emb(c_model, p_sent)  # [num_candidates, embedding_dim]
        with torch.no_grad():
            sim = torch.mm(it_centrid_embedding, p_emb.T)  # [1, num_candidates]
            temp_scores = sim.mean(dim=0)  # [num_candidates]

        candidate_scores += temp_scores
        end_time = time.time()
        # print(end_time - start_time, 'seconds one hot flipping')

    # print(current_score, max(candidate_scores).cpu().item())
    # print(current_acc_rate, max(candidate_acc_rates).cpu().item())

    return current_score, candidate_scores