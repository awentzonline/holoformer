import torch

from holoformer.models import hrr


def hrr_xml_loss(pred, target, embedding, p, m):
    pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target_embs = embedding(target)
    all_embs = embedding.weight.sum(0)
    neg_embs = all_embs - target_embs
    neg_embs = neg_embs / (torch.norm(neg_embs, dim=-1, keepdim=True) + 1e-8)
    m_pred = hrr.unbind(pred, m)
    cos_absent = torch.einsum('bi,bi->b', m_pred, neg_embs)
    J_n = torch.mean(torch.abs(cos_absent))

    target_embs = target_embs / (torch.norm(target_embs, dim=-1, keepdim=True) + 1e-8)
    p_pred = hrr.unbind(pred, p)
    cos_present = torch.einsum('bi,bi->b', p_pred, target_embs)
    J_p = torch.mean(1 - torch.abs(cos_present))
    return J_p, J_n
