# Localstack

You could check the following link to learn more about Localstack.
https://github.com/localstack/localstack

# Installation

We will run Localstack with Docker-compose. We are following the instructions given [here](https://docs.localstack.cloud/get-started/#docker-compose).
You could use the following commands to run the Localstack on Docker.

```bash
docker-compose -f localstack/docker-compose.yml up 
```

Now, we will create a bucket named `fssd-models`,and upload our model repository, that will be used in Triton Inference Server.
```bash
aws --endpoint-url=http://localhost:4566 s3api create-bucket --bucket fssd-models
aws --endpoint-url=http://localhost:4566 s3 cp model_repository/ s3://fssd-models/model_repository --recursive
```
Let's take a look at the files under `fssd-models` bucket.
```bash
aws --endpoint-url=http://localhost:4566 s3 ls s3://fssd-models
```
You should see the following output.
```bash
                           PRE model_repository/
```