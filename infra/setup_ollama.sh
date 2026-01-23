#!/bin/bash

# Atualiza o sistema e instala dependências
echo "Atualizando sistema e instalando curl..."
sudo apt update && sudo apt upgrade -y
sudo apt install curl ufw -y

# Permite porta 11434 no firewall (para WSL/WSL2)
sudo ufw allow 11434/tcp
sudo ufw --dry-run reload  # Apenas reload se já configurado

# Instala Ollama (oficial)
echo "Instalando Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Cria usuário e grupo ollama (recomendado para serviço)
sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama || true
sudo usermod -a -G ollama $USER

# Configura serviço systemd para rodar em background, exposto em 0.0.0.0:11434
sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
EOF

# Recarrega e inicia serviço
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Aguarda serviço iniciar
sleep 5
sudo systemctl status ollama --no-pager -l

# Baixa o modelo glm4:9b-chat-q5_0 (pode demorar ~10-30min dependendo da conexão)
echo "Baixando modelo glm4:9b-chat-q5_0 (~6.6GB)..."
ollama pull glm4:9b-chat-q5_0

# Testa funcionalidade
echo "Testando modelo..."
ollama run glm4:9b-chat-q5_0 "Olá, teste concluído!" || echo "Teste falhou, verifique logs com journalctl -u ollama -f"

echo "Instalação concluída! Ollama rodando em http://localhost:11434"
echo "Ver logs: journalctl -u ollama -f"
echo "Listar modelos: ollama list"
