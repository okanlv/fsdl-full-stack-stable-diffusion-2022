# Kubernetes

A good tutorial on Kubernetes basics could be found in the following link.
https://kubernetes.io/docs/tutorials/kubernetes-basics/

# Installation

Install `kubectl` and `minikube` by following the instructions on the following page

https://kubernetes.io/docs/tasks/tools/

The following instructions could be used to install minikube

## minikube installation

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

# Cluster configuration

- Start the cluster
```bash
minikube start
```

- Take a look at all the running pods in all the namespaces
```bash
kubectl get po -A
```

- You could also take a look at the Kubernetes Dashboard with the following command
```bash
minikube dashboard
```

- Create a sample deployment
```bash
kubectl create deployment hello-minikube --image=k8s.gcr.io/echoserver:1.4
kubectl expose deployment hello-minikube --type=NodePort --port=8080
```

- Browse the catalog of easily installed Kubernetes services:
```bash
minikube addons list
```

We need a have spare GPUs that are not used on the host and can be passthrough to the VM. It's not possible in our case.
https://minikube.sigs.k8s.io/docs/tutorials/nvidia_gpu/

- Delete the Minikube VM:
```bash
minikube delete
```