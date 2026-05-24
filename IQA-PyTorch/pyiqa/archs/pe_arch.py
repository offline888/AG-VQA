import decord
import perception_models.core.vision_encoder.pe as pe
import perception_models.core.vision_encoder.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from pyiqa.utils.registry import ARCH_REGISTRY

model_lists = {
    "PE-Core-B16-224": "/home/dzc/yuanhao/model/PE-Core-B16-224/PE-Core-B16-224.pt",
    "PE-Core-L14-336": "/home/dzc/yuanhao/model/PE-Core-L14-336/PE-Core-L14-336.pt",
}


def preprocess_video(
    video_path, num_frames=8, transform=None, return_first_frame_for_demo=True
):
    # Load the video
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    # Uniformly sample frame indices
    frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).asnumpy()
    # Preprocess frames
    preprocessed_frames = [transform(Image.fromarray(frame)) for frame in frames]

    first_frame = None
    if return_first_frame_for_demo:
        first_frame = frames[0]
    return torch.stack(preprocessed_frames, dim=0), first_frame

@ARCH_REGISTRY.register()
class PEVQA_Baseline(nn.Module):
    def __init__(self, backbone: str, device: str = "cuda"):
        super().__init__()
        model = pe.CLIP.from_config(
            backbone, pretrained=True, checkpoint_path=model_lists[backbone]
        )
        self.device = device
        self.model = model.to(device)
        self.model.train()

        self.preprocess = transforms.get_image_transform(model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(model.context_length)

        text_prompts = self.tokenizer(
            [
                "A video of bad quality",
                "A video of poor quality",
                "A video of fair quality",
                "A video of good quality",
                "A video of perfect quality",
            ]
        ).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.register_buffer("text_features", text_features)

        self.quality_weights = nn.Parameter(
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        return self

    def forward(self, videos, num_frames=8):
        if isinstance(videos, torch.Tensor):
            batch_video_tensor = videos.to(self.device)
        else:
            if isinstance(videos, str):
                videos = [videos]

            video_tensor_list = []
            for video_path in videos:
                v_tensor, _ = preprocess_video(
                    video_path, num_frames, transform=self.preprocess
                )
                video_tensor_list.append(v_tensor)

            # [B, T, C, H, W]
            batch_video_tensor = torch.stack(video_tensor_list, dim=0).to(self.device)

        # for trainging
        if self.training:
            video_features = self.model.encode_video(batch_video_tensor, normalize=True)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        else:
            # for inference
            with torch.no_grad():
                video_features = self.model.encode_video(
                    batch_video_tensor, normalize=True
                )
                video_features = video_features / video_features.norm(
                    dim=-1, keepdim=True
                )

        logits = 100.0 * (video_features @ self.text_features.T)
        probs = logits.softmax(dim=-1)
        score = (probs * self.quality_weights).sum(dim=-1)

        return {
            "quality_score": score,
            "quality_probs": probs,
            "video_features": video_features,
            "text_features": self.text_features,
            "logits": logits,
        }

@ARCH_REGISTRY.register()
class PEVQA_ProbText(nn.Module):
    def __init__(self, backbone: str, device: str = "cuda"):
        super().__init__()
        model = pe.CLIP.from_config(
            backbone, pretrained=True, checkpoint_path=model_lists[backbone]
        )
        self.device = device
        self.model = model.to(device)
        self.model.train()

        self.preprocess = transforms.get_image_transform(model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(model.context_length)

        text_prompts = self.tokenizer(
            [
                "A video of bad quality",
                "A video of poor quality",
                "A video of fair quality",
                "A video of good quality",
                "A video of perfect quality",
            ]
        ).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.register_buffer("text_mu", text_features)

        embed_dim = text_features.shape[-1]

        self.text_logvar = nn.Parameter(torch.zeros(5, embed_dim))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "quality_weights",
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        return self

    def forward(self, videos, num_frames=8):
        if isinstance(videos, str):
            videos = [videos]

        video_tensor_list = []
        for video_path in videos:
            v_tensor, _ = preprocess_video(
                video_path, num_frames, transform=self.preprocess
            )
            video_tensor_list.append(v_tensor)

        batch_video_tensor = torch.stack(video_tensor_list, dim=0).to(self.device)

        if self.training:
            mu_v = self.model.encode_video(batch_video_tensor, normalize=True)
            mu_v = mu_v / mu_v.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                mu_v = self.model.encode_video(batch_video_tensor, normalize=True)
                mu_v = mu_v / mu_v.norm(dim=-1, keepdim=True)

        mu_t = self.text_mu
        var_t = torch.exp(self.text_logvar)

        mu_v_ext = mu_v.unsqueeze(1)
        mu_t_ext = mu_t.unsqueeze(0)
        var_t_ext = var_t.unsqueeze(0)

        mean_diff = torch.sum((mu_v_ext - mu_t_ext) ** 2, dim=-1)
        std_diff = torch.sum(var_t_ext, dim=-1)

        w2_dist = mean_diff + std_diff

        logits = -self.gamma * w2_dist
        probs = logits.softmax(dim=-1)
        score = (probs * self.quality_weights).sum(dim=-1)

        return {
            "quality_score": score,
            "quality_probs": probs,
            "video_mean": mu_v,
            "text_mean": mu_t,
            "text_var": var_t,
            "w2_dist": w2_dist,
        }

@ARCH_REGISTRY.register()
class PEVQA_FeatDistribution(nn.Module):
    def __init__(self, backbone: str, device: str = "cuda"):
        super().__init__()
        model = pe.CLIP.from_config(
            backbone, pretrained=True, checkpoint_path=model_lists[backbone]
        )
        self.device = device
        self.model = model.to(device)
        self.model.train()

        self.preprocess = transforms.get_image_transform(model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(model.context_length)

        text_prompts = self.tokenizer(
            [
                "A video of bad quality",
                "A video of poor quality",
                "A video of fair quality",
                "A video of good quality",
                "A video of perfect quality",
            ]
        ).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.register_buffer("text_mu", text_features)

        embed_dim = text_features.shape[-1]

        self.text_logvar = nn.Parameter(torch.zeros(5, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.video_mean_head = nn.Linear(embed_dim, embed_dim)
        self.video_var_head = nn.Linear(embed_dim, embed_dim)

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "quality_weights",
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        return self

    def forward(self, videos, num_frames=8):
        if isinstance(videos, str):
            videos = [videos]

        video_tensor_list = []
        for video_path in videos:
            v_tensor, _ = preprocess_video(
                video_path, num_frames, transform=self.preprocess
            )
            video_tensor_list.append(v_tensor)

        batch_video_tensor = torch.stack(video_tensor_list, dim=0).to(self.device)
        B, T, C, H, W = batch_video_tensor.shape
        frames = batch_video_tensor.view(B * T, C, H, W)

        if self.training:
            frame_features = self.model.encode_image(frames, normalize=True)
        else:
            with torch.no_grad():
                frame_features = self.model.encode_image(frames, normalize=True)

        # (B, T, D)
        frame_features = frame_features.view(B, T, -1)

        temporal_feats = self.temporal_encoder(frame_features)
        global_feats = temporal_feats.mean(dim=1)

        mu_v = self.video_mean_head(global_feats)
        mu_v = mu_v / mu_v.norm(dim=-1, keepdim=True)
        var_v = F.softplus(self.video_var_head(global_feats))

        mu_t = self.text_mu
        var_t = torch.exp(self.text_logvar)

        mu_v_ext = mu_v.unsqueeze(1)
        var_v_ext = var_v.unsqueeze(1)
        mu_t_ext = mu_t.unsqueeze(0)
        var_t_ext = var_t.unsqueeze(0)

        mean_diff = torch.sum((mu_v_ext - mu_t_ext) ** 2, dim=-1)
        std_diff = torch.sum(
            (torch.sqrt(var_v_ext + 1e-8) - torch.sqrt(var_t_ext + 1e-8)) ** 2, dim=-1
        )

        w2_dist = mean_diff + std_diff

        logits = -self.gamma * w2_dist
        probs = logits.softmax(dim=-1)
        score = (probs * self.quality_weights).sum(dim=-1)

        return {
            "quality_score": score,
            "quality_probs": probs,
            "video_mean": mu_v,
            "video_var": var_v,
            "text_mean": mu_t,
            "text_var": var_t,
            "w2_dist": w2_dist,
        }
