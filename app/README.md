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
Now you can access Styreamlit app on your local machine on [http://localhost:8501](http://localhost:8501). In order to make it available via internet we use ngrok.

<div align="center"><h2>Share local Streamlit app with ngrok</h2></div>

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
    Forwarding                    https://aeda-188-119-39-165.ngrok.io -> http://localhost:8501  
   ```