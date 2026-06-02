# 使用国内镜像源加速 python:3.10-slim
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.10-slim

# 设置 pip 源为清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

# 安装系统依赖
# python:3.10-slim 基于 Debian，切换到国内 apt 镜像以减少首次构建等待时间。
RUN sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    pkg-config \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# gunicorn 和 prometheus-flask-exporter 已统一写入 requirements.txt，此处无需重复安装
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 预创建运行期目录，避免首次挂载时目录不存在
RUN mkdir -p /app/data/raw /app/data/processed /app/models/trained /app/chroma_db

# 暴露端口 (Flask/Gunicorn)
EXPOSE 5000
