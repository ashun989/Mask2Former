import numpy as np
from detectron2.data import transforms as T
from skimage.filters import gaussian


class CopyPasteTransform(T.Transform):
    def __init__(self,
                 alpha,
                 paste_img,
                 paste_seg,
                 blend=True,
                 sigma=3):
        super(CopyPasteTransform, self).__init__()
        self.alpha = alpha
        self.paste_img = paste_img
        self.paste_seg = paste_seg
        self.blend = blend
        self.sigma = sigma

    def apply_image(self, img: np.ndarray):
        alpha = gaussian(self.alpha, sigma=self.sigma, preserve_range=True) if self.blend else self.alpha
        img_dtype = img.dtype
        alpha = alpha[..., None]
        img = self.paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)
        return img

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        alpha = self.alpha
        segmentation = self.paste_seg * alpha + segmentation * (1 - alpha)
        return segmentation

    def apply_coords(self, coords: np.ndarray):
        raise NotImplementedError
