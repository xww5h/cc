download all files or do 'git clone' command

run docker build at the downloaded file folder, for example: `docker build -t ssndetect .`, where 'ssndetect' is the docker image name

  if you don't want to down the model file in docker (thus a large docker image), comment out line 38 of Dockerfile (wget -O /app/models/Qwen3-8B-Q8_0.gguf "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf?download=true")

run the container with mounted volumn (if you did not download model file in docker), for example: `docker run -it -p 7860:7860 -v ./models:/app/models ssndetect`

in the docker console, start the webserver, for example: `python app.py --model_path models/Qwen3-8B-Q8_0.gguf --nothink`

  try play with parameter of '--nothink' or without it to disable/enable 'think' mode (inference time computing)
