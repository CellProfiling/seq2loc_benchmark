import torch


def collate_fn(batch):
    embeddings, masks, locations = zip(*batch)
    embeddings = torch.stack(embeddings, dim=0)
    masks = torch.stack(masks, dim=0)
    locations = torch.stack(locations, dim=0)
    return embeddings, masks, locations


def test_collate_fn(batch):
    ids, seqs, embeddings, masks, locations = zip(*batch)

    ids = [i for i in ids]
    seqs = [s for s in seqs]
    embeddings = torch.stack(embeddings, dim=0)
    masks = torch.stack(masks, dim=0)
    locations = torch.stack(locations, dim=0)
    return ids, seqs, embeddings, masks, locations
