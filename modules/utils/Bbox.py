import torch

def bboxIntersection(bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [N,M,2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min = 0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    return inter