#!/bin/bash
# 清除整個 local k8s 環境
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${YELLOW}警告：這將刪除 rust-assistant cluster 的所有資料${NC}"
read -p "確定要繼續嗎？(y/N): " confirm
[[ "$confirm" == "y" || "$confirm" == "Y" ]] || { echo "已取消"; exit 0; }

k3d cluster delete rust-assistant
echo -e "${GREEN}Cluster 已刪除${NC}"