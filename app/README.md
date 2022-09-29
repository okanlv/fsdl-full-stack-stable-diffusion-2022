<div align="center"><h1>FSDL Full Stack Stable Diffusion Web Server</h1></div>

<div align="center"><h2>Download models</h2></div>

```bash
# clone this repo
git clone git@github.com:okanlv/fsdl-full-stack-stable-diffusion-2022.git
# download models
cd fsdl-full-stack-stable-diffusion-2022
sh triton/download_models.sh
```

<div align="center"><h2>Build Triton Inference Server Docker Image</h2></div>

```bash
# navigate to triton subdirectory
cd triton
# build triton server docker image
docker build -t tritonserver .
```

<div align="center"><h2>Build Streamlit Docker image</h2></div>

```bash
# inside app subdirectory run
docker build -t streamlit .
```

<div align="center"><h2>Run Docker Compose</h2></div>
This command will run triton inference server and streamlit docker containers.

```bash
docker compose up
```

<div align="center"><h2>Open Streamlit app</h2></div>

Streamlit web app URL:

[http://localhost:8501](http://localhost:8501)