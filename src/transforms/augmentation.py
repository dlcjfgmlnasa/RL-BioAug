# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F


class WeakAugmenter(object):
    def __init__(self, sigma=0.01, scale=0.02):
        self.sigma = sigma
        self.scale = scale

    def __call__(self, x):
        device = x.device
        out = x.clone()
        scaler = torch.empty(1, device=device).uniform_(1 - self.scale, 1 + self.scale)
        out = out * scaler
        noise = torch.randn_like(out, device=device) * self.sigma
        out = out + noise
        return out


class StrongAugmenter(object):
    def __init__(self):
        self.action_names = [
            'time_masking',
            'time_permutation',
            'crop_resize',
            'time_flip',
            'time_warp'
        ]

    def __call__(self, x, action_idx):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        c, t = x.shape
        device = x.device

        if action_idx == 0:     # [Action 1] Time Masking (Zero-out)
            mask_len = t // 4
            if mask_len == 0: return x

            start = torch.randint(0, t - mask_len, (1,), device=device).item()
            out = x.clone()
            out[:, start: start + mask_len] = 0
            return out

        elif action_idx == 1:   # [Action 2] Time Permutation (Shuffle Segments)
            n_permutation = 5
            valid_range = t - 2
            x_3d = x.unsqueeze(0)

            cut_points = torch.randperm(valid_range, device=device)[:n_permutation - 1] + 1
            cut_points = torch.cat([
                torch.tensor([0], device=device),
                cut_points,
                torch.tensor([t], device=device)
            ])
            cut_points, _ = torch.sort(cut_points)

            segments = []
            for idx_1, idx_2 in zip(cut_points[:-1], cut_points[1:]):
                segments.append(x_3d[:, :, idx_1:idx_2])

            perm_indices = torch.randperm(n_permutation, device=device)
            shuffled_segments = [segments[idx] for idx in perm_indices]

            x_split = torch.cat(shuffled_segments, dim=-1)
            return x_split.squeeze(0)

        elif action_idx == 2:   # [Action 3] Crop & Resize
            x_3d = x.unsqueeze(0)
            min_len = int(t * 0.3)
            max_len = int(t * 0.9)

            if min_len >= max_len: return x

            segment_size = torch.randint(min_len, max_len, (1,), device=device).item()
            max_start_idx = t - segment_size

            if max_start_idx <= 0: return x

            index_1 = torch.randint(0, max_start_idx, (1,), device=device).item()
            index_2 = index_1 + segment_size
            x_split = x_3d[:, :, index_1:index_2]  # (1, 1, Seg_Len)
            x_resized = F.interpolate(
                x_split,
                size=t,
                mode='linear',
                align_corners=False
            )
            return x_resized.squeeze(0)

        elif action_idx == 3:   # [Action 4] Time Flip
            return torch.flip(x, dims=[-1])

        elif action_idx == 4:   # [Action 5] Time Warp
            n_segments = 5
            x_3d = x.unsqueeze(0)
            segment_len = t // n_segments
            warp_ratios = torch.rand(n_segments, device=device).softmax(dim=0)

            warp_lens = (warp_ratios * t).long()
            warp_lens[-1] = t - warp_lens[:-1].sum()
            warped_segments = []
            curr_idx = 0
            for i in range(n_segments):
                end_idx = curr_idx + segment_len if i < n_segments - 1 else t
                seg = x_3d[:, :, curr_idx:end_idx]

                curr_idx = end_idx
                target_len = max(1, warp_lens[i].item())

                if seg.shape[-1] > 0:
                    warped_seg = F.interpolate(
                        seg, size=target_len, mode='linear', align_corners=False
                    )
                    warped_segments.append(warped_seg)

            if len(warped_segments) > 0:
                x_warped = torch.cat(warped_segments, dim=-1)
                if x_warped.shape[-1] != t:
                    x_warped = F.interpolate(x_warped, size=t, mode='linear', align_corners=False)
                return x_warped.squeeze(0)
            else:
                return x

        return x
