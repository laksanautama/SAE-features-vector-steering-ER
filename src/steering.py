"""
Steering context managers for hook-based activation modification.
"""

import torch
import torch.nn.functional as F


class MultiLayerSteeringContext:
    """
    Registers pre-hooks at multiple layers. Per-sample emotion-conditioned steering.
    Used when alpha is (n_emotions_at_layer, n_cand) per layer.
    """

    def __init__(self, model, alpha_dict, V_dict, unique_layers, layer_emotion_map,
                 emotion_to_idx, n_layers, layer_ids=None, emo_ids=None):
        self.model = model
        self.alpha_dict = alpha_dict
        self.V_dict = V_dict
        self.unique_layers = unique_layers
        self.layer_emotion_map = layer_emotion_map
        self.emotion_to_idx = emotion_to_idx
        self.n_layers = n_layers
        self.layer_ids = layer_ids
        self.emo_ids = emo_ids
        self.handles = []

    def __enter__(self):
        for ul_idx, layer_j in enumerate(self.unique_layers):
            nxt = layer_j + 1
            target = (self.model.model.layers[nxt] if nxt < self.n_layers
                      else self.model.model.norm)
            self.handles.append(
                target.register_forward_pre_hook(self._make_hook(ul_idx, layer_j)))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, ul_idx, layer_j):
        emos_at = self.layer_emotion_map[layer_j]
        emo_gids = [self.emotion_to_idx[e] for e in emos_at]
        alpha = self.alpha_dict[ul_idx]
        V = self.V_dict[ul_idx]
        emo_ids = self.emo_ids

        def hook_fn(module, args):
            h = args[0]
            bs = h.size(0)
            delta = torch.zeros(bs, h.size(-1), device=h.device, dtype=h.dtype)

            for local_idx, emo_gid in enumerate(emo_gids):
                mask = (emo_ids == emo_gid)
                if mask.sum() == 0:
                    continue
                alpha_pos = F.softplus(alpha[local_idx]).to(dtype=V.dtype)
                d_emo = torch.matmul(alpha_pos, V).to(dtype=h.dtype)
                delta[mask] = d_emo

            return (h + delta.unsqueeze(1),) + args[1:]

        return hook_fn


class SharedSteeringContext:
    """
    Registers pre-hooks at multiple layers. Same delta for all samples per layer.
    Used when alpha is (n_cand,) per layer.
    """

    def __init__(self, model, alpha_dict, V_dict, unique_layers, n_layers):
        self.model = model
        self.alpha_dict = alpha_dict
        self.V_dict = V_dict
        self.unique_layers = unique_layers
        self.n_layers = n_layers
        self.handles = []

    def __enter__(self):
        for ul_idx, layer_j in enumerate(self.unique_layers):
            nxt = layer_j + 1
            target = (self.model.model.layers[nxt] if nxt < self.n_layers
                      else self.model.model.norm)
            self.handles.append(
                target.register_forward_pre_hook(self._make_hook(ul_idx)))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, ul_idx):
        alpha = self.alpha_dict[ul_idx]
        V = self.V_dict[ul_idx]

        def hook_fn(module, args):
            h = args[0]
            alpha_pos = F.softplus(alpha).to(dtype=V.dtype)
            delta = torch.matmul(alpha_pos, V).to(dtype=h.dtype)
            return (h + delta.unsqueeze(0).unsqueeze(0),) + args[1:]

        return hook_fn


class SelectiveSteeringContext:
    """
    Steers at only a subset of layers (for ablation studies).
    """

    def __init__(self, model, alpha_dict, V_dict, unique_layers, active_layers,
                 n_layers, emo_ids=None, layer_emotion_map=None, emotion_to_idx=None):
        self.model = model
        self.alpha_dict = alpha_dict
        self.V_dict = V_dict
        self.unique_layers = unique_layers
        self.active_layers = active_layers
        self.n_layers = n_layers
        self.emo_ids = emo_ids
        self.layer_emotion_map = layer_emotion_map
        self.emotion_to_idx = emotion_to_idx
        self.handles = []

    def __enter__(self):
        for ul_idx, layer_j in enumerate(self.unique_layers):
            if layer_j not in self.active_layers:
                continue
            nxt = layer_j + 1
            target = (self.model.model.layers[nxt] if nxt < self.n_layers
                      else self.model.model.norm)
            self.handles.append(
                target.register_forward_pre_hook(self._make_hook(ul_idx, layer_j)))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, ul_idx, layer_j):
        alpha = self.alpha_dict[ul_idx]
        V = self.V_dict[ul_idx]

        if alpha.dim() == 1:
            # Shared alpha
            def hook_fn(module, args):
                h = args[0]
                ap = F.softplus(alpha).to(dtype=V.dtype)
                delta = torch.matmul(ap, V).to(dtype=h.dtype)
                return (h + delta.unsqueeze(0).unsqueeze(0),) + args[1:]
        else:
            # Emotion-conditioned alpha
            emos_at = self.layer_emotion_map[layer_j]
            emo_gids = [self.emotion_to_idx[e] for e in emos_at]
            emo_ids = self.emo_ids

            def hook_fn(module, args):
                h = args[0]
                delta = torch.zeros(h.size(0), h.size(-1), device=h.device, dtype=h.dtype)
                for li, eid in enumerate(emo_gids):
                    m = (emo_ids == eid)
                    if m.sum() == 0:
                        continue
                    delta[m] = torch.matmul(
                        F.softplus(alpha[li]).to(dtype=V.dtype), V
                    ).to(dtype=h.dtype)
                return (h + delta.unsqueeze(1),) + args[1:]

        return hook_fn
