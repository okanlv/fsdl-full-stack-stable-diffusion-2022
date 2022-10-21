#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This module is copy-pasted in generated Triton configuration folder to perform inference.
"""

import inspect
import io


# noinspection DuplicatedCode
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image
import torch
from transformers import CLIPTokenizer
from diffusers.schedulers import DDIMScheduler, PNDMScheduler
from diffusers.models.vae import DiagonalGaussianDistribution

try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.


class TritonPythonModel:
    tokenizer: CLIPTokenizer
    device: str
    scheduler: Union[DDIMScheduler, PNDMScheduler]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    eta: float


    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        current_name: str = str(Path(args["model_repository"]).parent.absolute())
        self.device = "cpu" if args["model_instance_kind"] == "CPU" else "cuda"
        self.tokenizer = CLIPTokenizer.from_pretrained(
            current_name + "/stable_diffusion/1/"
        )
        self.scheduler = PNDMScheduler()
        self.scheduler = self.scheduler.set_format("pt")
        self.height = 512
        self.width = 512
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.eta = 0.0
        self.strength = 0.75

    def preprocess(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image.contiguous() - 1.0

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "PROMPT")
                .as_numpy()
                .tolist()
            ]
            init_image = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "INIT_IMAGE")
                .as_numpy()
                .tolist()
            ][0]
            batch_size = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "SAMPLES")
                .as_numpy()
                .tolist()
            ][0]
            self.num_inference_steps = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "STEPS")
                .as_numpy()
                .tolist()
            ][0]
            self.guidance_scale = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "GUIDANCE_SCALE")
                .as_numpy()
                .tolist()
            ][0]
            seed = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "SEED")
                .as_numpy()
                .tolist()
            ][0]

            prompt = prompt * batch_size
            # get prompt text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = text_input.input_ids.type(dtype=torch.int64)
            inputs = [pb_utils.Tensor.from_dlpack("tokens", torch.to_dlpack(input_ids))]

            inference_request = pb_utils.InferenceRequest(
                model_name="encoder",
                requested_output_names=["text_embeddings"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "text_embeddings"
                )
                text_embeddings: torch.Tensor = torch.from_dlpack(output.to_dlpack())

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = self.guidance_scale > 1.0
            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                max_length = text_input.input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    [""] * batch_size,
                    padding="max_length",
                    max_length=max_length,
                    truncation=False,
                    return_tensors="pt",
                )

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                input_ids = uncond_input.input_ids.type(dtype=torch.int64)
                inputs = [
                    pb_utils.Tensor.from_dlpack("tokens", torch.to_dlpack(input_ids))
                ]

                inference_request = pb_utils.InferenceRequest(
                    model_name="encoder",
                    requested_output_names=["text_embeddings"],
                    inputs=inputs,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(
                        inference_response, "text_embeddings"
                    )
                    uncond_embeddings: torch.Tensor = torch.from_dlpack(
                        output.to_dlpack()
                    )

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            generator = torch.Generator(device=self.device).manual_seed(seed)

            # set timesteps
            accepts_offset = "offset" in set(
                inspect.signature(self.scheduler.set_timesteps).parameters.keys()
            )
            extra_set_kwargs = {}
            offset = 0
            if accepts_offset:
                offset = 1
                extra_set_kwargs["offset"] = 1

            self.scheduler.set_timesteps(self.num_inference_steps, **extra_set_kwargs)

            init_image = self.preprocess(init_image)
            init_image.to(self.device)
            inputs = [
                pb_utils.Tensor.from_dlpack("init_image", torch.to_dlpack(init_image))
            ]
            inference_request = pb_utils.InferenceRequest(
                model_name="img_encoder",
                requested_output_names=["init_image_latents"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(inference_response.error().message())
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "init_image_latents"
                )
                init_image_latents: torch.Tensor = torch.from_dlpack(output.to_dlpack())

            inputs = [
                pb_utils.Tensor.from_dlpack(
                    "init_image_latents", torch.to_dlpack(init_image_latents)
                )
            ]

            inference_request = pb_utils.InferenceRequest(
                model_name="quant_conv",
                requested_output_names=["init_latents"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(inference_response.error().message())
            else:
                output = pb_utils.get_output_tensor_by_name(inference_response, "init_latents")
                init_latents: torch.Tensor = torch.from_dlpack(output.to_dlpack())

            diagonal_gaussian_dist = DiagonalGaussianDistribution(init_latents)
            init_latents = diagonal_gaussian_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

            # expand init_latents for batch_size
            init_latents = torch.cat([init_latents] * batch_size)

            # get the original timestep using init_timestep
            init_timestep = int(self.num_inference_steps * self.strength) + offset
            init_timestep = min(init_timestep, self.num_inference_steps)
            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

            # add noise to latents using the timesteps
            noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
            init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)


            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = self.eta
            latents = init_latents
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = latent_model_input.type(dtype=torch.float32)
                timestep = t[None].type(dtype=torch.int64)
                encoder_hidden_states = text_embeddings.type(dtype=torch.float32)

                inputs = [
                    pb_utils.Tensor.from_dlpack(
                        "sample", torch.to_dlpack(latent_model_input)
                    ),
                    pb_utils.Tensor.from_dlpack("timestep", torch.to_dlpack(timestep)),
                    pb_utils.Tensor.from_dlpack(
                        "encoder_hidden_states", torch.to_dlpack(encoder_hidden_states)
                    ),
                ]

                inference_request = pb_utils.InferenceRequest(
                    model_name="unet",
                    requested_output_names=["noise_pred"],
                    inputs=inputs,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(
                        inference_response, "noise_pred"
                    )
                    noise_pred: torch.Tensor = torch.from_dlpack(output.to_dlpack())

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents

            inputs = [pb_utils.Tensor.from_dlpack("latents", torch.to_dlpack(latents))]
            inference_request = pb_utils.InferenceRequest(
                model_name="post_quant_conv",
                requested_output_names=["post_quant_latents"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "post_quant_latents"
                )
                post_quant_latents: torch.Tensor = torch.from_dlpack(output.to_dlpack())

            inputs = [
                pb_utils.Tensor.from_dlpack(
                    "latents", torch.to_dlpack(post_quant_latents)
                )
            ]
            inference_request = pb_utils.InferenceRequest(
                model_name="decoder",
                requested_output_names=["image"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(inference_response, "image")
                image: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()

            tensor_output = [pb_utils.Tensor("IMAGES", image)]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
