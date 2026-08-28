from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torchvision.transforms as T
import torchreid


class FeatureExtractor:
    """
    Torchreid feature extractor

    The model is never trained on the current target person.
    We rely on the pretrained weights provided by torchreid.
    It only produces embeddings for the online gallery.
    """

    def __init__(self,
                 device=None,
                 model_name: str = "osnet_x1_0",
                 weights_path: Optional[Union[str, Path]] = None, ):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device(device)
        self.model_name = str(model_name)

        # Torchreid loads the library-provided pretrained initialization.
        try:
            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=751,
                pretrained=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not build torchreid model {self.model_name!r}. "
                "Use a valid architecture such as 'osnet_x1_0' 'osnet_x0_75', "
                "'osnet_x0_5', 'osnet_x0_25', or 'resnet50'."
            ) from exc

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization, resized to the ReID input size.
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Keep track where the weights come from.
        self.weight_source = "torchreid pretrained=True (library-provided architecture weights)"

        # Count for the PerformanceLogger to see how often ReID is triggered
        self.extract_calls = 0

    def extract(self, img):
        """Extract feature embedding from an image crop."""
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            feat = self.model(img_tensor)

        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        result = feat.squeeze(0).detach().cpu().numpy()

        self.extract_calls += 1

        return result
