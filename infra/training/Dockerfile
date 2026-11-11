ARG PYTHON_BASE_IMAGE=python@sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /opt/program
COPY src/waymark/remote_requirements.txt /opt/program/remote_requirements.txt
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends gcc libc6-dev libgomp1 \
    && pip install --no-cache-dir -r /opt/program/remote_requirements.txt \
       'sagemaker-training @ https://files.pythonhosted.org/packages/source/s/sagemaker-training/sagemaker_training-5.1.4.tar.gz#sha256=4b6147670ccee4b14cbd86dbfdf416fcfffe486b3ef7ac8d1caf6bc361386121' \
    && pip check \
    && apt-get purge -y --auto-remove gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    SAGEMAKER_PROGRAM=train_entry.py \
    AWS_DEFAULT_REGION=ap-southeast-2
ENTRYPOINT ["train"]
