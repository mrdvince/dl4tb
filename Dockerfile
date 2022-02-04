from python:3.9-slim

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install torch==1.10.2+cpu torchvision==0.11.3+cpu  -f https://download.pytorch.org/whl/cpu/torch_stable.html

COPY ./inf_requirements.txt /tmp/inf_requirements.txt

RUN pip install -r /tmp/inf_requirements.txt

RUN pip install "dvc[s3]"

COPY . /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY


# aws credentials configuration
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d storage s3://dvc-store123/trained_models
# pulling the trained model
RUN dvc pull dvcfiles/onnx_model.dvc

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]