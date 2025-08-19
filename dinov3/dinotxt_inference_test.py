import matplotlib.pyplot as plt
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
    dinotxt_weights="/Users/mbenisch/Downloads/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    backbone_weights="/Users/mbenisch/Downloads/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)

import urllib
from PIL import Image

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
img_pil = load_image_from_url(EXAMPLE_IMAGE_URL)
img_pil.show()

import torch
from dinov3.data.transforms import make_classification_eval_transform

image_preprocess = make_classification_eval_transform()
image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).cpu()
texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
class_names = ["dog", "chair", "bowl", "tupperware"]
tokenized_texts_tensor = tokenizer.tokenize(texts).cpu()
model = model.cpu()
with torch.autocast('cpu', dtype=torch.float):
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(tokenized_texts_tensor)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (
    text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
)
print(similarity)

with torch.autocast('cpu', dtype=torch.float):
    with torch.no_grad():
        image_class_tokens, image_patch_tokens, backbone_patch_tokens = model.encode_image_with_patch_tokens(image_tensor)
        text_features_aligned_to_patch = model.encode_text(tokenized_texts_tensor)[:, 1024:] # Part of text features that is aligned to patch features

import torch.nn.functional as F

B, P, D = image_patch_tokens.shape
H = W = int(P**0.5)
x = image_patch_tokens.movedim(2, 1).unflatten(2, (H, W)).float()  # [B, D, H, W]
x = F.interpolate(x, size=(480, 640), mode="bicubic", align_corners=False)
x = F.normalize(x, p=2, dim=1)
y = F.normalize(text_features_aligned_to_patch.float(), p=2, dim=1)
per_patch_similarity_to_text = torch.einsum("bdhw,cd->bchw", x, y)
pred_idx = per_patch_similarity_to_text.argmax(1).squeeze(0)
pred_idx_np = pred_idx.numpy()
plt.imshow(pred_idx_np)
plt.show()