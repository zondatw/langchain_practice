#!/bin/bash
# ─────────────────────────────────────────────
# Rust Assistant - Local k8s 環境建立腳本
# 工具：k3d + kubectl + helm
# ─────────────────────────────────────────────
set -e

# 取得 script 所在目錄，讓相對路徑正確運作
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 1. 確認依賴 ──────────────────────────────
info "檢查依賴工具..."
for cmd in docker kubectl helm; do
  command -v $cmd &>/dev/null || error "$cmd 未安裝，請先安裝後再執行"
done

# 安裝 k3d（如果還沒裝）
if ! command -v k3d &>/dev/null; then
  info "安裝 k3d..."
  brew install k3d 2>/dev/null || curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
fi

# ── 2. 建立 cluster ───────────────────────────
CLUSTER_NAME="rust-assistant"
if k3d cluster list | grep -q "$CLUSTER_NAME"; then
  warning "Cluster '$CLUSTER_NAME' 已存在，跳過建立"
else
  info "建立 k3d cluster..."
  k3d cluster create "$CLUSTER_NAME" \
    --port "80:80@loadbalancer" \
    --port "443:443@loadbalancer" \
    --port "6333:6333@loadbalancer" \
    --k3s-arg "--disable=traefik@server:0" \
    --agents 1
fi

kubectl config use-context "k3d-$CLUSTER_NAME"
info "Cluster 已就緒，目前 context: $(kubectl config current-context)"

# ── 3. 加入 Helm repos ────────────────────────
info "加入 Helm repos..."
helm repo add qdrant       https://qdrant.github.io/qdrant-helm        2>/dev/null || true
helm repo add prometheus   https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add grafana      https://grafana.github.io/helm-charts        2>/dev/null || true
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx  2>/dev/null || true
helm repo update

# ── 4. 建立 namespaces ────────────────────────
info "建立 namespaces..."
kubectl apply -f "$SCRIPT_DIR/namespaces.yaml"

# ── 5. 部署各元件 ─────────────────────────────
info "部署 Prometheus + Grafana（需先安裝，Qdrant ServiceMonitor 依賴其 CRD）..."
helm upgrade --install kube-prometheus-stack prometheus/kube-prometheus-stack \
  -n monitoring \
  -f "$SCRIPT_DIR/monitoring/prometheus-values.yaml" \
  --wait --timeout 180s

info "部署 Qdrant..."
helm upgrade --install qdrant qdrant/qdrant \
  -n qdrant \
  -f "$SCRIPT_DIR/qdrant/values.yaml" \
  --wait --timeout 120s

info "啟用 Qdrant ServiceMonitor（Prometheus CRD 已就緒）..."
helm upgrade qdrant qdrant/qdrant \
  -n qdrant \
  -f "$SCRIPT_DIR/qdrant/values.yaml" \
  --set metrics.serviceMonitor.enabled=true \
  --set metrics.serviceMonitor.namespace=monitoring \
  --set metrics.serviceMonitor.interval=15s \
  --wait --timeout 60s

info "部署 Ingress Nginx..."
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  -n ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer \
  --wait --timeout 120s

# ── 6. 套用 Ingress 規則 ──────────────────────
info "套用 Ingress 規則..."
kubectl apply -f "$SCRIPT_DIR/ingress/ingress.yaml"

info "套用 Qdrant ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/qdrant/configmap.yaml"

info "套用 Grafana Dashboard ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/monitoring/grafana-dashboard-cm.yaml"

# ── 7. 完成 ───────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  環境建立完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Qdrant     : http://localhost:6333"
echo "  Qdrant UI  : http://qdrant.rust-assistant.local      (需設定 /etc/hosts)"
echo "  Grafana    : http://grafana.rust-assistant.local      (admin / admin)"
echo "  Prometheus : http://prometheus.rust-assistant.local"
echo ""
echo "  快速設定 /etc/hosts："
echo "  echo '127.0.0.1 qdrant.rust-assistant.local grafana.rust-assistant.local prometheus.rust-assistant.local' | sudo tee -a /etc/hosts"
echo ""
echo "  查看所有 pods："
echo "  kubectl get pods -A"