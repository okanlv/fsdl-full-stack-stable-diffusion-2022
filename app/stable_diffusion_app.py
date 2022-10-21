# -*- coding: utf-8 -*-
import os
import io

import numpy as np
import PIL
import streamlit as st
import tritonclient.http
from PIL import Image, ImageOps


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class SDTritonServer:
    def __init__(self, model_name, url, model_version, batch_size) -> None:
        # Input placeholder
        self.prompt_in = tritonclient.http.InferInput(
            name="PROMPT", shape=(batch_size,), datatype="BYTES"
        )
        self.init_image_in = tritonclient.http.InferInput(
            name="INIT_IMAGE", shape=(batch_size,), datatype="BYTES"
        )
        self.samples_in = tritonclient.http.InferInput(
            "SAMPLES", (batch_size,), "INT32"
        )
        self.steps_in = tritonclient.http.InferInput("STEPS", (batch_size,), "INT32")
        self.guidance_scale_in = tritonclient.http.InferInput(
            "GUIDANCE_SCALE", (batch_size,), "FP32"
        )
        self.seed_in = tritonclient.http.InferInput("SEED", (batch_size,), "INT64")
        self.images = tritonclient.http.InferRequestedOutput(
            name="IMAGES", binary_data=False
        )

        # triton client
        self.triton_client = tritonclient.http.InferenceServerClient(
            url=url, verbose=False
        )
        assert self.triton_client.is_model_ready(
            model_name=model_name, model_version=model_version
        ), f"model {model_name} not yet ready"

        self.model_metadata = self.triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version
        )
        self.model_config = self.triton_client.get_model_config(
            model_name=model_name, model_version=model_version
        )

    def infer(
        self,
        prompt,
        samples,
        steps,
        guidance_scale,
        seed,
        init_image=None,
    ):
        # Setting inputs
        self.prompt_in.set_data_from_numpy(
            np.asarray([prompt] * batch_size, dtype=object)
        )
        self.samples_in.set_data_from_numpy(np.asarray([samples], dtype=np.int32))
        self.steps_in.set_data_from_numpy(np.asarray([steps], dtype=np.int32))
        self.guidance_scale_in.set_data_from_numpy(
            np.asarray([guidance_scale], dtype=np.float32)
        )
        self.seed_in.set_data_from_numpy(np.asarray([seed], dtype=np.int64))

        if init_image is not None:
            self.init_image_in.set_data_from_numpy(
                np.asarray([init_image] * batch_size, dtype=object)
            )

            # do inference
            response = self.triton_client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=[
                    self.prompt_in,
                    self.init_image_in,
                    self.samples_in,
                    self.steps_in,
                    self.guidance_scale_in,
                    self.seed_in,
                ],
                outputs=[self.images],
                timeout=1200000,
            )
        else:
            # do inference
            response = self.triton_client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=[
                    self.prompt_in,
                    self.samples_in,
                    self.steps_in,
                    self.guidance_scale_in,
                    self.seed_in,
                ],
                outputs=[self.images],
                timeout=1200000,
            )
        # get generated images
        images = response.as_numpy("IMAGES")
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


if __name__ == "__main__":
    hide_st_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    # model args
    model_name = "stable_diffusion_txt2img"
    url = "inference_server:8000"
    model_version = "1"
    batch_size = 1

    # model input params
    samples = 1
    seed = 1024

    if "grid_images" not in st.session_state:
        st.session_state.grid_images = None

    path = os.path.dirname(__file__)
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.sidebar.title("About")
    st.sidebar.info(
        "**Stable Diffusion App** can Generate photo-realistic images based on your input prompt\n\n"
        "**Text to Image**:\n\n"
        "You should enter a prompt, choose number of steps and guidance scale and click generate.\n\n"
        "**Image to Image**:\n\n"
        "You should enter a prompt, upload an initial image and choose number of steps and guidance scale and click generate.\n\n"
    )
    Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
    </style> 
    """
    st.markdown(Title_html, unsafe_allow_html=True)  # Title rendering
    st.header("Stable Diffusion App")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        prompt = st.text_input(
            "Prompt",
            value="",
            key="prompt",
            placeholder="The quick brown fox on mars",
        )
        # TODO: Investigate tritonserver timeout error when using larger step size > 50
        steps = st.slider(
            label="Number of Steps: ", min_value=1, max_value=500, value=50, key="steps"
        )
        guidance_scale = st.slider(
            label="Guidance Scale: ",
            min_value=1.0,
            max_value=20.0,
            step=0.5,
            value=7.5,
            key="guidance_scale",
        )

        st.write(f"steps: {steps}, guidance_scale: {guidance_scale}")

    with col2:
        uploaded_image = st.file_uploader(
            "Upload an Image (Optional)", type=["jpg", "jpeg", "png"]
        )
        if uploaded_image is not None:
            init_image = Image.open(uploaded_image)
            st.image(init_image, caption="Uploaded Image.", use_column_width=True)
            is_init_image = True
            model_name = "stable_diffusion"
        else:
            image_placeholder = st.image(f"{path}/assets/placeholder.png")
            image_placeholder.empty()
            is_init_image = False
        st.text(body="Sample Generated Image")
        generated_image = st.empty()
        if st.session_state.grid_images is None:
            generated_image.image(
                f"{path}/assets/the_quick_brown_fox_on_mars.png",
                caption="Generated Image",
            )
        else:
            generated_image.image(
                st.session_state.grid_images, caption="Generated Image"
            )
    if st.button("Generate Image"):
        st.success(f"Generating image/images for prompt: {prompt}!")
        sd_triton_server = SDTritonServer(
            model_name=model_name,
            url=url,
            model_version=model_version,
            batch_size=batch_size,
        )
        if is_init_image is True:
            buf = io.BytesIO()
            init_image.save(buf, format="png")
            init_image = buf.getvalue()
            pil_images = sd_triton_server.infer(
                prompt=prompt,
                samples=samples,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                init_image=init_image,
            )
        else:
            pil_images = sd_triton_server.infer(
                prompt=prompt,
                samples=samples,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        with col2:
            st.session_state.grid_images = image_grid(pil_images, rows=1, cols=1)
            generated_image.image(
                st.session_state.grid_images, caption="Generated Image"
            )
