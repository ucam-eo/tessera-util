#!/bin/bash

# Daintree
REMOTE_USER=zf281
REMOTE_HOST=daintree.cl.cam.ac.uk
REMOTE_DIR=/scratch/zf281/data_processed
LOCAL_TMP_DIR=/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/tiles_tmp
LOCAL_DIR=/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/tiles

# Step 1: 从远程获取有效目录（子文件夹完整）
rm -f valid_dirs_raw.txt valid_dirs.txt

echo "📥 正在从远程服务器筛选包含完整 .npy 文件的子文件夹..."

ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && \
for d in */; do
  files='band_mean.npy bands.npy band_std.npy doys.npy masks.npy sar_ascending_doy.npy sar_ascending.npy sar_descending_doy.npy sar_descending.npy'
  all_exist=true
  for f in \$files; do
    if [ ! -f \"\$d\$f\" ]; then
      all_exist=false
      break
    fi
  done
  if \$all_exist; then
    echo \"\$d\"
  fi
done" > valid_dirs_raw.txt

total_remote_dirs=$(wc -l < valid_dirs_raw.txt)
echo "✅ 远程完整目录共 ${total_remote_dirs} 个，正在检查本地是否已存在..."

# Step 2: 本地筛查哪些子目录尚未存在或不完整
touch valid_dirs.txt
while read dir; do
    dirname=$(echo "$dir" | tr -d '/')  # 去除最后的斜杠
    local_path="${LOCAL_DIR}/${dirname}"
    
    # 定义需要检查的文件列表
    files=("band_mean.npy" "bands.npy" "band_std.npy" "doys.npy" "masks.npy" "sar_ascending_doy.npy" "sar_ascending.npy" "sar_descending_doy.npy" "sar_descending.npy")
    
    # 检查本地目录是否存在且完整
    if [ -d "$local_path" ]; then
        all_present=true
        for f in "${files[@]}"; do
            if [ ! -f "${local_path}/${f}" ]; then
                all_present=false
                break
            fi
        done
        if $all_present; then
            echo "✅ 本地已有完整目录：$dirname，跳过下载。"
            continue
        else
            echo "⚠️ 本地目录 $dirname 不完整，需要重新下载。"
        fi
    else
        echo "🆕 本地不存在目录：$dirname，将下载。"
    fi
    
    # 将需要下载的目录添加到列表
    echo "$dir" >> valid_dirs.txt
done < valid_dirs_raw.txt

needed_dirs=$(wc -l < valid_dirs.txt)
echo "📦 需要下载的目录数：${needed_dirs} / ${total_remote_dirs}"

# Step 3: 下载所需子目录
if [ "$needed_dirs" -eq 0 ]; then
    echo "🎉 所有目录已存在且完整，无需下载。"
    exit 0
fi

# 确保目标目录存在
mkdir -p ${LOCAL_TMP_DIR}

# 开始下载
while read dir; do
    dirname=$(echo "$dir" | tr -d '/')
    echo "🔄 正在复制目录: $dirname (${dir})"
    
    # 确保目标子目录存在
    mkdir -p "${LOCAL_TMP_DIR}/${dirname}"
    
    rsync -avz --progress --info=progress2 \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${dir}"* \
        "${LOCAL_TMP_DIR}/${dirname}/"
    
    # 检查传输是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 目录 $dirname 传输完成"
    else
        echo "❌ 目录 $dirname 传输失败"
    fi
done < valid_dirs.txt

echo "🎉 所有需要的目录传输任务已完成！"

# Dawn
# REMOTE_USER=zf281
# REMOTE_HOST=login-icelake.hpc.cam.ac.uk
# REMOTE_DIR=/home/zf281/rds/rds-sj514-data-WBrUDmBgqOo/s2_s1_global_project/data_processed
# LOCAL_TMP_DIR=/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/tiles_tmp
# LOCAL_DIR=/media/12TBNVME/frankfeng/btfm4rs/data/ssl_training/tiles

# # Step 1: 从远程获取有效目录（子文件夹完整）
# rm -f valid_dirs_raw.txt valid_dirs.txt

# echo "📥 正在从远程服务器筛选包含完整 .npy 文件的子文件夹..."

# ssh -o ControlPath=~/.ssh/cm_socket ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && \
# for d in */; do
#   files='band_mean.npy bands.npy band_std.npy doys.npy masks.npy sar_ascending_doy.npy sar_ascending.npy sar_descending_doy.npy sar_descending.npy'
#   all_exist=true
#   for f in \$files; do
#     if [ ! -f \"\$d\$f\" ]; then
#       all_exist=false
#       break
#     fi
#   done
#   if \$all_exist; then
#     echo \"\$d\"
#   fi
# done" > valid_dirs_raw.txt

# total_remote_dirs=$(wc -l < valid_dirs_raw.txt)
# echo "✅ 远程完整目录共 ${total_remote_dirs} 个，正在检查本地是否已存在..."

# # Step 2: 本地筛查哪些子目录尚未存在或不完整
# touch valid_dirs.txt
# while read dir; do
#     dirname=$(echo "$dir" | tr -d '/')  # 去除最后的斜杠
#     local_path="${LOCAL_DIR}/${dirname}"
    
#     # 定义需要检查的文件列表
#     files=("band_mean.npy" "bands.npy" "band_std.npy" "doys.npy" "masks.npy" "sar_ascending_doy.npy" "sar_ascending.npy" "sar_descending_doy.npy" "sar_descending.npy")
    
#     # 检查本地目录是否存在且完整
#     if [ -d "$local_path" ]; then
#         all_present=true
#         for f in "${files[@]}"; do
#             if [ ! -f "${local_path}/${f}" ]; then
#                 all_present=false
#                 break
#             fi
#         done
#         if $all_present; then
#             echo "✅ 本地已有完整目录：$dirname，跳过下载。"
#             continue
#         else
#             echo "⚠️ 本地目录 $dirname 不完整，需要重新下载。"
#         fi
#     else
#         echo "🆕 本地不存在目录：$dirname，将下载。"
#     fi
    
#     # 将需要下载的目录添加到列表
#     echo "$dir" >> valid_dirs.txt
# done < valid_dirs_raw.txt

# needed_dirs=$(wc -l < valid_dirs.txt)
# echo "📦 需要下载的目录数：${needed_dirs} / ${total_remote_dirs}"

# # Step 3: 下载所需子目录
# if [ "$needed_dirs" -eq 0 ]; then
#     echo "🎉 所有目录已存在且完整，无需下载。"
#     exit 0
# fi

# # 确保目标目录存在
# mkdir -p ${LOCAL_TMP_DIR}

# # 开始下载
# while read dir; do
#     dirname=$(echo "$dir" | tr -d '/')
#     echo "🔄 正在复制目录: $dirname (${dir})"
    
#     # 确保目标子目录存在
#     mkdir -p "${LOCAL_TMP_DIR}/${dirname}"
    
#     rsync -avz --progress --info=progress2 \
#         -e "ssh -o ControlPath=~/.ssh/cm_socket" \
#         "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${dir}"* \
#         "${LOCAL_TMP_DIR}/${dirname}/"
    
#     # 检查传输是否成功
#     if [ $? -eq 0 ]; then
#         echo "✅ 目录 $dirname 传输完成"
#     else
#         echo "❌ 目录 $dirname 传输失败"
#     fi
# done < valid_dirs.txt

# echo "🎉 所有需要的目录传输任务已完成！"