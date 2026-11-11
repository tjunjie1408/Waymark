ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
WORKDIR /opt/program
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY src/waymark/remote_requirements.txt /opt/program/waymark/remote_requirements.txt
RUN pip install --no-cache-dir -r /opt/program/waymark/remote_requirements.txt
COPY src/waymark/__init__.py src/waymark/batch_inference.py src/waymark/batch_contract.py src/waymark/feature_schema.py src/waymark/training_environment.py src/waymark/artifacts.py /opt/program/waymark/
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["python", "-m", "waymark.batch_inference"]
