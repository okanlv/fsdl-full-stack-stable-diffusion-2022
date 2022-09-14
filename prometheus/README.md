# Prometheus

A good tutorial on Prometheus could be found in the following link.
https://prometheus.io/docs/prometheus/latest/getting_started/

# Installation

We will use run Prometheus on Docker. You could check the following link to learn more about it.
https://hub.docker.com/r/prom/prometheus

You could use the following command to pull and run the latest Prometheus image. This image will use the `prometheus.yml`
file to configure the Prometheus. Note that you should run this command from the main repository directory.


```bash
docker run \
    -p 9090:9090 \
    -v $(pwd)/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```

You could check the Prometheus status page at http://localhost:9090. You could stop the docker container afterward.


Now, we will do the followings to get used to Prometheus.
- Install the official Python client for Prometheus from [here](https://github.com/prometheus/client_python)

    First, we will create and activate a conda environment. Then, we will install the Prometheus client.
```bash
conda create -n fssd python=3.10 && conda activate fssd
pip install prometheus-client==0.14.1
```

- Run an example script, taken from the same repository, and log some metrics to an endpoint.

```bash
python prometheus/dummy_server.py
```

- Check the metrics from the endpoint.

    You could check the dummy server metric page at http://localhost:8000/.

- Configure `prometheus.yml` file so that Prometheus could scrape these metrics.

    Add the following job definition to the scrape_configs section in the `prometheus.yml`. We have already done that.
```yaml
scrape_configs:
  - job_name: 'dummy_server'

    static_configs:
      - targets: ['localhost:8000']
```

- Run the Prometheus on Docker.

    Open another terminal. We will run Prometheus on Docker, but this time we pass `--net host` parameters as well.
    This will give Prometheus access to the host network, so that it could scrape the metrics from the localhost.

```bash
docker run --net host\
    -p 9090:9090 \
    -v $(pwd)/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```

- Check the metrics on the Prometheus status page.

    Open the Prometheus status page at http://localhost:9090. You could execute the following query to the average (dummy)
    request duration per second.

```bash
rate(dummy_request_processing_seconds_sum[5m]) / rate(dummy_request_processing_seconds_count[5m])
```

- You could stop the dummy server and the docker container afterward.
