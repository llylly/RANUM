"""
    Clip the scale, span, center defined ranges by the range defined in initial_abstracts
"""
import torch

EPS = 1e-5

def range_clipping(initial_abstracts: dict, centers: dict, scales: dict, spans: dict):

    def work(init_lb, init_ub, name):
        if name not in scales:
            # only handle centers is sufficient
            centers[name] = torch.minimum(torch.maximum(centers[name], init_lb), init_ub)
        else:
            clipped_min = torch.maximum(init_lb, centers[name] - scales[name] * spans[name])
            clipped_max = torch.minimum(init_ub, centers[name] + scales[name] * spans[name])
            new_center = (clipped_min + clipped_max) / 2.
            new_scale = torch.where((spans[name] <= EPS) & (spans[name] >= -EPS), torch.zeros_like(new_center), (new_center - clipped_min) / spans[name])
            centers[name] = new_center
            scales[name] = new_scale

    for abst_name, abst_obj in initial_abstracts.items():
        if isinstance(abst_obj.lb, list):
            for i in range(len(abst_obj.lb)):
                work(abst_obj.lb[i], abst_obj.ub[i], abst_name + f'_{i}')
        else:
            work(abst_obj.lb, abst_obj.ub, abst_name)
