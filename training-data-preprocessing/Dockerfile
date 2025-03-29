# 第一阶段: 构建
FROM rust:1.74-slim as builder

# 安装依赖(如 libssl-dev) 仅示例
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY Cargo.toml .
COPY src ./src

RUN cargo build --release

# 第二阶段: 运行
FROM debian:buster-slim

RUN apt-get update && apt-get install -y \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/app/target/release/training-data-preprocessing /usr/local/bin/

ENTRYPOINT ["training-data-preprocessing"]

# docker build -t training-data-prep .
# docker create --name temp-copy training-data-prep:latest
# docker cp temp-copy:/usr/local/bin/training-data-preprocessing ./
# docker rm temp-copy

# ./training-data-preprocessing \
#   --data-root /maps/zf281/btfm-training-10.4/data_processed/global \
#   --output-dir /maps/zf281/btfm-training-10.4/data_processed/ready_to_use \
#   --tile-batch 100 \
#   --time-steps 20 \
#   --chunk-size 10000 \
#   --min-valid-timesteps 20

# RUST_LOG=debug ./training-data-preprocessing \
#   --data-root /maps/zf281/btfm-training-10.4/data_processed/global \
#   --output-dir /maps/zf281/btfm-training-10.4/data_processed/ready_to_use \
#   --tile-batch 130 \
#   --time-steps 20 \
#   --chunk-size 10000 \
#   --min-valid-timesteps 20
