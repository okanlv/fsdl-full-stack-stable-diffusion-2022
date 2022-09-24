# fsdl-full-stack-stable-diffusion-2022

You need to run the following services.

## Localstack

Run Localstack with Docker-compose using the following commands.
```bash
docker-compose -f localstack/docker-compose.yml up 
```

In a separate terminal, create a bucket named `fssd-models`,and upload our model repository, that will be used in Triton Inference Server.
```bash
aws --endpoint-url=http://localhost:4566 s3api create-bucket --bucket fssd-models
aws --endpoint-url=http://localhost:4566 s3 cp triton/sd-v1-4-onnx/models s3://fssd-models/model_repository --recursive
```

## Triton Inference Server

Build the custom Triton Inference Server docker image

```bash
docker build -t tritonserver triton/.
```

Run the Triton Inference Server with the custom image.

```bash
docker run -it --rm --gpus all --net=host --shm-size 16384m \
-v $PWD/triton/sd-v1-4-onnx-test/models:/models tritonserver tritonserver --model-repository /models/
```

You could check the Triton Inference Server metric page at http://localhost:8002/metrics.

Note that the following command also works but you need to change `model.py` for `stable_diffusion`. Change the following
line
```python
self.tokenizer = CLIPTokenizer.from_pretrained(current_name + "/stable_diffusion/1/")
```
to
```python
self.tokenizer = CLIPTokenizer.from_pretrained(args["model_repository"] + "/1/")
```

```bash
docker run --gpus=1 --rm --net=host  \
      -e AWS_ACCESS_KEY_ID='dummy_id' -e AWS_SECRET_ACCESS_KEY='dummy_key'  \
      tritonserver tritonserver \
      --model-repository=s3://localhost:4566/fssd-models/model_repository
```

## Prometheus

Run Prometheus using the following command.

```bash
docker run --net host\
    -p 9090:9090 \
    -v $(pwd)/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```
## Grafana

In a separate terminal, run Grafana using the following command..
```bash
# create a persistent volume for your data in /var/lib/grafana (database and plugins)
docker volume create grafana-storage

# start grafana
docker stop grafana && docker rm grafana
docker run -d --net host\
  -p 3000:3000 \
  -v grafana-storage:/var/lib/grafana \
  -v $(pwd)/grafana/dashboards:/etc/dashboards \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning \
  --name grafana \
  -e "GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-piechart-panel" \
  grafana/grafana-oss
```

You could check the Grafana page at http://localhost:3000. Enter `admin` for the username and password. Then, 
go to http://localhost:3000/d/S-BvSNn4k/triton-inference-server-dashboard page to see the Triton Inference Server Dashboard.
Take a look at this dashboard after you have send a few request in the following section.

## Streamlit

Create a python environment to run Streamlit

```bash
conda create --name webserver python=3.10
conda activate webserver
pip install -r app/requirements.txt
```

Run the web server.

```bash
cd app 
streamlit run stable_diffusion_app.py
```

You can now view your Streamlit app in http://localhost:8501.