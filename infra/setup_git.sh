#!/usr/bin/env bash
set -e

REPO_SSH_URL="git@github.com:AirtonLira/dspy_ai_learning.git"
REPO_HTTP_URL="https://github.com/AirtonLira/dspy_ai_learning.git"

echo "=== 0) Indo para a raiz do repo (se estiver em infra/) ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Diretório atual: $(pwd)"
echo

echo "=== 1) Validando repositório Git ==="
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Erro: este diretório não é um repositório Git."
  echo "Execute: git init  && git add . && git commit -m \"initial commit\""
  exit 1
fi
echo "Repositório Git OK."
echo

echo "=== 2) Garantindo branch 'main' ==="
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
  echo "Branch atual: $CURRENT_BRANCH -> renomeando para 'main'..."
  git branch -M main
else
  echo "Branch já é 'main'."
fi
echo

echo "=== 3) Limpando helpers de credencial antigos ==="
echo "Removendo helpers em system/global/local (se existirem)..."
sudo git config --system --unset credential.helper 2>/dev/null || true
git config --global --unset credential.helper 2>/dev/null || true
git config --local  --unset credential.helper 2>/dev/null || true

echo "Removendo ~/.git-credentials (se existir)..."
rm "$HOME/.git-credentials" 2>/dev/null || true
echo "Helpers e credenciais limpos."
echo

echo "=== 4) Configurando remote 'origin' para SSH ==="
if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote 'origin' já existe. Atualizando para SSH:"
  echo "  $REPO_SSH_URL"
  git remote set-url origin "$REPO_SSH_URL"
else
  echo "Criando remote 'origin' com SSH:"
  echo "  $REPO_SSH_URL"
  git remote add origin "$REPO_SSH_URL"
fi
echo

echo "=== 5) Criando chave SSH (se não existir) ==="
SSH_KEY="$HOME/.ssh/id_ed25519"
if [ -f "$SSH_KEY" ]; then
  echo "Chave SSH já existe em: $SSH_KEY"
else
  echo "Gerando nova chave SSH em: $SSH_KEY"
  mkdir -p "$HOME/.ssh"
  ssh-keygen -t ed25519 -C "airtonlirajr@gmail.com" -f "$SSH_KEY" -N ""
fi
echo

echo "=== 6) Mostrando chave pública (cole no GitHub) ==="
echo
echo "COPIE A CHAVE ABAIXO E COLE EM:"
echo "  GitHub -> Settings -> SSH and GPG keys -> New SSH key"
echo "-------------------------------------------------------"
cat "$SSH_KEY.pub"
echo "-------------------------------------------------------"
echo
echo "Após colar e salvar a chave no GitHub, volte aqui e pressione ENTER para continuar."
read -r _

echo "=== 7) Testando autenticação SSH com GitHub ==="
ssh -T git@github.com || true
echo
echo "Se a mensagem acima disser algo como:"
echo "  \"Hi AirtonLira! You've successfully authenticated, but GitHub does not provide shell access.\""
echo "então a autenticação está OK."
echo

echo "=== 8) Garantindo que há pelo menos um commit ==="
if ! git log -1 >/dev/null 2>&1; then
  echo "Nenhum commit encontrado. Criando commit inicial..."
  git add .
  git commit -m "chore: initial commit"
else
  echo "Já existe pelo menos um commit."
fi
echo

echo "=== 9) Fazendo push para 'origin main' via SSH ==="
git push -u origin main

echo
echo "=============================================="
echo "Push concluído via SSH para: $REPO_SSH_URL"
echo "Daqui pra frente, use normalmente:"
echo "  git add ..."
echo "  git commit -m \"mensagem\""
echo "  git push"
echo "=============================================="
