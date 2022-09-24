# Triton Inference Server

In this part, we will run a toy example in Triton Inference Server, and show the collected metrics in Grafana. You could 
check the following link to learn more about it.
https://github.com/triton-inference-server/server

# Prerequisites

Make sure that your Nvidia-Driver supports Nvidia 22.02 container image. Check the following documentation to see the supported versions.
https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

You could check the following page to download the supported driver.
https://www.nvidia.com/Download/index.aspx?lang=en-us

# Installation

We will run Triton Inference Server on Docker. We are following the instructions given [here](https://github.com/triton-inference-server/server#serve-a-model-in-3-easy-steps).
You could use the following commands to run the Triton Inference Server.

```bash
# get densenet_onnx model
wget -O model_repository/densenet_onnx/1/model.onnx \
     https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx

# launch triton from the NGC Triton container
docker run --gpus=1 --rm --net=host -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models
```

Note that if have followed the steps under 'localstack/README.md', you should have a running localstack image with model files inside a bucket. Instead of attaching a local
folder and use it to run the Triton Inference Server, as we have done above, you could directly download the files from AWS S3 localstack using the following command. You
could check out [here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md#model-repository-locations
) if you want to learn more about this.

```bash
docker run --gpus=1 --rm --net=host  \
      -e AWS_ACCESS_KEY_ID='dummy_id' -e AWS_SECRET_ACCESS_KEY='dummy_key'  \
      nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver \
      --model-repository=s3://localhost:4566/fssd-models/model_repository
```

Now, let's send a few request.
```bash
# In a separate console, launch the image_client example from the NGC Triton SDK container
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:22.02-py3-sdk
```

Send some requests by running the following line inside the container a few times. This command will send a request
to the Triton Inference Server every 1 second
```bash
watch -n1 /workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```

You should get a result similar to the following one
Send some requests by running the following line inside the container a few times
```bash
Image '/workspace/images/mug.jpg':
    15.349420 (504) = COFFEE MUG
    13.226885 (968) = CUP
    10.425241 (505) = COFFEEPOT
```

You could check the Triton Inference Server metric page at http://localhost:8002/metrics.

Now, in a separate terminal, run the Prometheus on Docker so that we could collect these metrics.

```bash
docker run --net host\
    -p 9090:9090 \
    -v $(pwd)/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```

In a separate terminal, run Grafana as well
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