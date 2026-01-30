#!/bin/bash

# Garantir que o script rode como root
if [ "$EUID" -ne 0 ]; then 
  echo "Por favor, execute como sudo."
  exit
fi

echo "--- Iniciando Otimização do Ubuntu 24.04 no WSL2 ---"

# 1. Atualização de Repositórios e Sistema
echo "Atualizando pacotes..."
apt update && apt upgrade -y

# 2. Instalação de utilitários essenciais e NVIDIA Toolkit
# Nota: O driver em si é instalado no Windows, aqui instalamos a ponte
echo "Configurando NVIDIA Container Toolkit..."
apt install -y ubuntu-drivers-common nvidia-utils-535 # Versão estável para 24.04

# 3. Configuração de Variáveis de Ambiente para Performance (CUDA)
echo "Configurando variáveis de ambiente no .bashrc..."
if ! grep -q "LD_LIBRARY_PATH" ~/.bashrc; then
  echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
fi

# 4. Ajuste de Limite de Arquivos (Útil para Treinamento de Modelos)
echo "Aumentando limites de sistema..."
echo "* soft nofile 65535" >> /etc/security/limits.conf
echo "* hard nofile 65535" >> /etc/security/limits.conf

# 5. Forçar persistência da GPU (ajuda na latência de inicialização)
nvidia-smi -pm 1

echo "--- Otimização Interna Concluída ---"
echo "Nota: Verifique o arquivo .wslconfig no Windows para performance de hardware."