FROM python:3.10-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels \
        numpy==1.26.4 h5py==3.11.0 scipy==1.13.1 \
        scikit-learn==1.5.1 tqdm==4.66.5

WORKDIR /build
COPY . .

RUN mkdir -p src/python/package/internal \
                 src/python/package/include \
                 src/python/package/external \
    && cp src/python/wrapper/python_wrapper.cc \
          src/python/package/internal/python_wrapper.cc \
    && cp -r src/include/. src/python/package/include/ \
    && cp -r external/eigen src/python/package/external/eigen \
    && cp -r external/pybind11 src/python/package/external/pybind11 \
    && cp -r external/simple-serializer \
             src/python/package/external/simple-serializer \
    && cp README.md src/python/package/README.md \
    && cp LICENSE.txt src/python/package/LICENSE.txt \
    && python -m pip wheel --no-cache-dir --no-deps --no-build-isolation \
         --wheel-dir /wheels ./src/python/package

FROM python:3.10-slim

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

WORKDIR /workspace
COPY experiments/run_experiment.py experiments/run_experiment.py

ENV DATASET_ROOT=/datasets \
    RESULTS_ROOT=/results \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "experiments/run_experiment.py"]
