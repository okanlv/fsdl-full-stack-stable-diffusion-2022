<div align="center"><h1>FSDL Full Stack Stable Diffusion Web Server</h1></div>

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

<div align="center"><h2>Install Web Server Requirements</h2></div>
In another terminal inside repo directory navigate to app directory and install web server requirements

```bash
cd app
conda create --name webserver python=3.10
conda activate webserver
pip install -r requirements.txt
```

<div align="center"><h2>Run webserver</h2></div>

Run the web server.

```bash
streamlit run stable_diffusion_app.py
```

You will see an output similar to the one below

```bash
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.102:8501
```
