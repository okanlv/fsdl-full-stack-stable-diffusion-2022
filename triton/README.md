<div align="center"><h1>FSDL Full Stack Stable Diffusion Triton Inference Server</h1></div>

<div align="center"><h2>Download models</h2></div>

```bash
# clone this repo
git clone git@github.com:okanlv/fsdl-full-stack-stable-diffusion-2022.git
# download models
cd fsdl-full-stack-stable-diffusion-2022
sh triton/download_models.sh
```

<div align="center"><h2>Build Triton Inference Server</h2></div>

```bash
# navigate to triton subdirectory
cd triton
# build triton server docker image
docker build -t tritonserver .
```

<div align="center"><h2>Run Triton Inference Server</h2></div>

```bash
# inside triton subdirectory run
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 16384m \
-v $PWD/sd-v1-4-onnx/models:/models tritonserver tritonserver --model-repository /models/
```
