# Grafana

A good tutorial on Prometheus could be found in the following link.
https://grafana.com/docs/grafana/latest/

# Installation

We will run Grafana on Docker. You could check the following link to learn more about it.
https://grafana.com/docs/grafana/latest/setup-grafana/installation/docker/
https://grafana.com/docs/grafana/latest/administration/provisioning/

You could use the following command to pull and run the latest Grafana image.

```bash
# create a persistent volume for your data in /var/lib/grafana (database and plugins)
docker volume create grafana-storage

# start grafana
docker run -d --net host\
  -p 3000:3000 \
  -v grafana-storage:/var/lib/grafana \
  -v $(pwd)/grafana/dashboards:/etc/dashboards \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning \
  --name grafana \
  -e "GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-piechart-panel" \
  grafana/grafana-oss
```

Note that
- "grafana/provisioning/datasources" folder keeps our data source configuration.
- "grafana/provisioning/dashboards" folder keeps our main dashboard configuration.
- "grafana/dashboards" folder keeps each dashboard configuration.

You could check the Grafana page at http://localhost:3000. Enter `admin` for the username and password.

https://grafana.com/docs/grafana/latest/setup-grafana/configure-docker/

Now, run the Prometheus on Docker so that we could take a look at some metrics.

```bash
docker run --net host\
    -p 9090:9090 \
    -v $(pwd)/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```

Also, run the example script to collect some extra metrics.

```bash
conda activate fssd
python prometheus/dummy_server.py
```

Then, go to http://localhost:3000/explore page to explore some Prometheus metrics. For example, you could execute the following query to get the average (dummy)
request duration per second that is being logged by "dummy_server.py" script.

```bash
rate(dummy_request_processing_seconds_sum[5m]) / rate(dummy_request_processing_seconds_count[5m])
```

You could also check out an example dashboard we have created on http://localhost:3000/dashboards. Click "General" folder, 
then select "Example dashboard". We show the same metric as before, the average (dummy) request duration per second that is 
being logged by "dummy_server.py" script.
