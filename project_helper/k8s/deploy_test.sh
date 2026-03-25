#!/bin/bash
set -e

# 1. 確認所有 pod 都是 Running
kubectl get pods -A

# 2. 確認 ingress controller 有跑起來
kubectl get pods -n ingress-nginx

# 3. 確認 ingress 規則有套用
kubectl get ingress -A

# 4. 確認 k3d loadbalancer port 有映射
k3d cluster list
docker ps | grep k3d