#!/bin/bash


for img in *.jpg *.jpeg *.png *.bmp *.gif; do
    if [ -f "$img" ]; then  # 检查文件是否存在
        # 使用 identify 读取图片分辨率
        resolution=$(identify -format "%wx%h" "$img")
        
        # 如果不是 1920x1080 分辨率，就删除该文件
        if [ "$resolution" != "1920x1080" ]; then
            echo "Deleting $img (Resolution: $resolution)"
            rm "$img"  # 删除文件
        else
            echo "Keeping $img (Resolution: $resolution)"
        fi
    fi
done