import torch


def hrr_xml_loss(pred, target, embedding):
    pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target_embs = embedding(target)
    target_embs = target_embs / (torch.norm(target_embs, dim=-1, keepdim=True) + 1e-8)
    all_embs = embedding.weight.sum()
    neg_embs = all_embs - target_embs
    neg_embs = neg_embs / (torch.norm(neg_embs, dim=-1, keepdim=True) + 1e-8)
    cos_absent = torch.einsum('bi,bi->b', pred, neg_embs)
    J_n = torch.mean(torch.abs(cos_absent))

    cos_present = torch.einsum('bi,bi->b', pred, target_embs)
    J_p = torch.mean(1 - torch.abs(cos_present))
    return J_p + J_n
