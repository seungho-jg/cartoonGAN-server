import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

class StableDiffusionCartoonizer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def cartoonize(self, image, prompt="anime style", strength=0.75, guidance_scale=7.5):
        # 이미지를 PIL Image로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 이미지 크기 조정 (메모리 사용량 감소를 위해)
        image = image.resize((512, 512))

        # 이미지 변환
        result = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale
        ).images[0]

        # NumPy 배열로 변환
        return np.array(result)

def load_model():
    return StableDiffusionCartoonizer()

def cartoon_gan_inference(model, input_image, style="anime"):
    prompt = f"{style} style"
    return model.cartoonize(input_image, prompt=prompt)