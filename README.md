# fsdl-full-stack-stable-diffusion-2022

You need to clone this repo and run the following services.
## Clone the repo and download models
```bash
# clone this repo
git clone git@github.com:okanlv/fsdl-full-stack-stable-diffusion-2022.git
# download models
cd fsdl-full-stack-stable-diffusion-2022
sh triton/download_models.sh
```


## Localstack

Run Localstack with Docker-compose using the following commands.
```bash
docker-compose -f localstack/docker-compose.yml up --detach
```

Now, create a bucket named `fssd-models`,and upload our model repository, that will be used in Triton Inference Server.
```bash
aws --endpoint-url=http://localhost:4566 s3api create-bucket --bucket fssd-models
aws --endpoint-url=http://localhost:4566 s3 cp triton/sd-v1-4-onnx/models s3://fssd-models/model_repository --recursive
```

## Run Triton Inference Server and Streamlit app

Build the custom Triton Inference Server docker image

```bash
docker build -t tritonserver triton/.
```

Build the custom Streamlit App docker image

```bash
docker build -t streamlit app/.
```

Run the Triton Inference Server and Streamlit app.

```bash
docker-compose -f app/docker-compose.yml up
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


## Share local Streamlit app with ngrok

1. Create a ngrok acount [here](https://dashboard.ngrok.com/signup) and log in.
2. Install ngrok
```bash
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && \
sudo apt update && sudo apt install ngrok
```

3. Authenticate your ngrok agent.
  ```bash
  ngrok authtoken <relace-with-your-Authtoken>
  ```
  you can get your token by visiting [here](https://dashboard.ngrok.com/get-started/your-authtoken).

4. Forward your local Streamlit app to ngrok

```bash
ngrok http 8501
```
Now you will see a link to your streamlit app via ngrok similar to the following which forward localhost:8501 to an ngrok url.

```bash
ngrok by @inconshreveable                                                                                                    

Session Status                online                                                                                                                                                                                                                                                         
Account                       Omid (Plan: Free)                                                                                                                                                                                                                                              
Version                       2.3.40                                                                                                                                                                                                                                                         
Region                        United States (us)                                                                                                                                                                                                                                             
Web Interface                 http://127.0.0.1:4040 
Forwarding                    http://aeda-188-119-39-165.ngrok.io -> http://localhost:8501                                                                                                                                                                                                   
```

You can now view your Streamlit app in the link provided by ngrok.