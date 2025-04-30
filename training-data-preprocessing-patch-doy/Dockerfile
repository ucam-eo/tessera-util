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

COPY --from=builder /usr/src/app/target/release/training-data-preprocessing-patch-doy /usr/local/bin/

ENTRYPOINT ["training-data-preprocessing-patch-doy"]

# docker build -t training-data-prep-patch-doy .
# docker create --name temp-copy training-data-prep-patch-doy:latest
# docker cp temp-copy:/usr/local/bin/training-data-preprocessing-patch-doy ./
# docker rm temp-copy


# RUST_LOG=debug /media/12TBNVME/frankfeng/btfm4rs/training-data-preprocessing-patch-doy/training-data-preprocessing-patch-doy --data
# -root /media/12TBNVME/frankfeng/btfm4rs/data/raw_tile_data/robin_fungal --chunk-size 5000 --patch-size 25 --max-patch-per-1k-1k-tile 5000 --stride 5 --tile-batch 1 --output-dir /media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/ready_to_use_patch