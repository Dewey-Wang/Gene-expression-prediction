# Dockerfile
FROM continuumio/miniconda3:latest
SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

# OS deps（其實用 conda 裝 pybigwig 不一定需要，但留著安全）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libcurl4-openssl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 先更新 conda
RUN conda update -n base -c defaults conda -y

# 關鍵：加入 bioconda，並保證 conda-forge 在最前面
RUN conda config --system --add channels conda-forge && \
    conda config --system --add channels bioconda && \
    conda config --system --set channel_priority strict

# 建環境並用 conda 安裝所有重型套件（含 pybigwig）
RUN conda create -n ml4g_project1 -y \
      python=3.12 \
      jupyterlab ipykernel \
      numpy pandas scipy scikit-learn \
      matplotlib seaborn graphviz python-graphviz \
      lightgbm xgboost catboost shap statsmodels umap-learn optuna \
      pybigwig \
    && conda clean -afy

# add near the end of your Dockerfile
SHELL ["/bin/bash", "-lc"]

# write a small launcher
RUN cat >/usr/local/bin/start-jupyter.sh <<'BASH'
#!/usr/bin/env bash
set -e
# pass through extra args (e.g., a working dir volume)
conda run -n ml4g_project1 jupyter lab \
  --ServerApp.ip=0.0.0.0 \
  --ServerApp.allow_remote_access=True \
  --ServerApp.open_browser=False \
  --ServerApp.token='' \
  --allow-root
BASH
RUN chmod +x /usr/local/bin/start-jupyter.sh

# default working dir (matches your volume mount)
WORKDIR /workspace

# default command
CMD ["/usr/local/bin/start-jupyter.sh"]
