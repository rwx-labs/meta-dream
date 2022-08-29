import inspect
import os
import warnings
import logging
import random
from typing import List, Optional, Union


from PIL import Image as img
from PIL.Image import Image
from tqdm.auto import tqdm
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers import DiffusionPipeline
from torch import autocast


# The model repo for Stable Diffusion.
STABLE_DIFFUSION_MODEL_ID = "CompVis/stable-diffusion-v1-4"
# The model repo for the CLIPTokenizer.
CLIP_TOKENIZER_MODEL_ID = "openai/clip-vit-large-patch14"
# The model repo used for the CLIPTextModel.
CLIP_TEXT_ENCODER_MODEL_ID = "openai/clip-vit-large-patch14"


class Sample:
    image: Image
    prompt: str
    seed: int

    def __init__(
        self, image: Image, prompt: str, seed: int = random.randint(0, 2**64)
    ):
        self.image = image
        self.prompt = prompt
        self.seed = seed


class StableDiffusionPipeline:
    def __init__(
        self,
        huggingface_token: str,
        model_cache_dir: Union[str, bytes, os.PathLike],
        torch_device: str = "cuda",
    ):
        self.device = torch_device
        self.logger = logging.getLogger(__name__)

        # Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained(
            STABLE_DIFFUSION_MODEL_ID,
            subfolder="vae",
            use_auth_token=huggingface_token,
            cache_dir=model_cache_dir,
        )

        self.logger.debug(f"Autoencoder: {vae.__class__.__name__}")

        # Load the tokenizer and text encoder to tokenize and encode the text.
        tokenizer = CLIPTokenizer.from_pretrained(
            CLIP_TOKENIZER_MODEL_ID, cache_dir=model_cache_dir
        )
        self.logger.debug(f"Tokenizer: {tokenizer.__class__.__name__}")

        text_encoder = CLIPTextModel.from_pretrained(
            CLIP_TEXT_ENCODER_MODEL_ID, cache_dir=model_cache_dir
        )
        assert isinstance(text_encoder, CLIPTextModel)
        self.logger.debug(f"Text encoder: {text_encoder.__class__.__name__}")

        # The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained(
            STABLE_DIFFUSION_MODEL_ID,
            subfolder="unet",
            use_auth_token=huggingface_token,
            cache_dir=model_cache_dir,
        )
        self.logger.debug(f"U-Net: {unet.__class__.__name__}")

        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.logger.debug(f"Scheduler: {self.scheduler.__class__.__name__}")

        # Move the models to the GPU
        self.vae = vae.to(torch_device)
        self.text_encoder = text_encoder.to(torch_device)
        self.unet = unet.to(torch_device)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        seed: int = random.randint(0, 2**64),
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        **kwargs,
    ):
        # Limit to a batch size of 1 until there's better batch handling in
        # place.
        batch_size = 1

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # Create a Generator with a deterministic seed
        generator = torch.Generator(self.device).manual_seed(seed or 0)

        print(f"tokenizer model max length: {self.tokenizer.model_max_length}")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=self.device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
            latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            assert latents

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, i, latents, **extra_step_kwargs
                )["prev_sample"]
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )["prev_sample"]

        # scale and decode the image latents with vae
        assert latents
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        assert output_type == "pil"

        sample = Sample(image[0], prompt, seed)

        return {"sample": sample}

    def scheduler_name(self):
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            return "k_lms"
        elif isinstance(self.scheduler, DDIMScheduler):
            return "ddim"
        elif isinstance(self.scheduler, PNDMScheduler):
            return "pndm"

        return None

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [img.fromarray(image) for image in images]

        return pil_images
