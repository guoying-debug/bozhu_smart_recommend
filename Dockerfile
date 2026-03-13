# 使用国内镜像源加速 python:3.10-slim
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.10-slim

# 设置 pip 源为清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# 安装系统依赖 (gcc/g++ 用于编译某些 Python 库, default-libmysqlclient-dev 用于 MySQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    default-libmysqlclient-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 增加 gunicorn 和 prometheus-client
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn prometheus-flask-exporter

# 复制项目代码
COPY . .

# 暴露端口 (Flask/Gunicorn)
EXPOSE 5000

# 启动命令 (使用 Gunicorn)
# 4个 worker 进程, 绑定 0.0.0.0:5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
