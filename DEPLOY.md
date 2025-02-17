# Documentação Deploy - Datathon

Este documento descreve o fluxo completo para construir, enviar e implantar a aplicação no AWS ECS (Fargate) utilizando o script build_and_push.sh.

## Requisitos do Ambiente Local

Certifique-se de que sua máquina possua os seguintes itens instalados e configurados:
- **Docker Desktop:** Necessário para construir e executar imagens Docker.
- **Git Bash (ou outro terminal Unix-like):** Recomendado para executar scripts Bash.
- **Credenciais AWS configuradas:** Um arquivo `.env` com suas credenciais.

Exemplo de `.env`:

```dotenv
AWS_REGION=
AWS_ACCOUNT_ID=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SESSION_TOKEN=

REPO_NAME=datathon-recommendation
IMAGE_TAG=latest

ECS_CLUSTER=datathon-cluster
ECS_SERVICE=datathon-service
SUBNETS=subnet-0550b1c6bf59381af,subnet-03d0e2edab32be5aa,subnet-09a96f864be9c8d2c,subnet-0232231a8a1937035,subnet-0555c752631e66e3c,subnet-095571597785d998f
SECURITY_GROUPS=sg-0219fea6259c18ecd
```

## Sumário

1. [Build, Push e Deploy no AWS ECS](#build-push-e-deploy-no-aws-ecs)
2. [Testando a API](#testando-a-api)
3. [Conclusão](#conclusão)

---

## Build, Push e Deploy no AWS ECS

O script `build_and_push.sh` executa todo o fluxo de automação, realizando as seguintes ações:

- Carrega as variáveis do arquivo `.env`.
- Faz login no AWS ECR.
- Verifica/cria o repositório no ECR.
- Constrói a imagem Docker utilizando o Dockerfile na pasta `docker/`.
- Tagueia e envia (push) a imagem para o ECR.
- Verifica/cria o cluster no ECS, gera uma nova Task Definition e registra uma nova revisão.
- Atualiza ou recria o serviço ECS para usar a nova Task Definition.
- (Opcional) Força a parada das tasks antigas e, após aguardar, exibe o IP público da nova task.

### Passos para executar:

1. **Abra o Docker Desktop** e, em seguida, abra o Git Bash na pasta raiz do projeto.

2. Torne o script executável (se necessário):

   ```bash
   chmod +x build_and_push.sh
   ```

3. Execute o script:

   ```bash
   ./build_and_push.sh
   ```

O script exibirá mensagens informando o progresso de cada etapa, incluindo a criação/atualização do serviço e a obtenção do IP público da task.

---

## Testando a API

Após o deploy, aguarde alguns instantes até que a nova task esteja em execução. Em seguida:

1. **Obtenha o IP Público da Task:**
   - O script exibirá o IP público no final da execução. Anote esse endereço.


2. **Teste com curl:**

   ```bash
   curl -v -X POST http://<IP_PUBLICO>:5000/recommendation \
        -H "Content-Type: application/json" \
        -d '{"user_id": "seu_userId_valido", "history": []}'
   ```

3. **Teste com Postman:**

   **Method:** POST  
   **URL:** `http://<IP_PUBLICO>:5000/recommendation`

   **Headers:**
     - **Key:** `Content-Type`  
       **Value:** `application/json`

   **Body (raw, JSON):**
   ```json
   {
     "user_id": "c33bcbcaf32fe895fb0a854c94531120657ad5adabe3e14b57fc4b4167074906",
     "history": []
   }
   ```

---

## Conclusão

1. Execute o script `build_and_push.sh` para construir a imagem Docker, enviá-la para o ECR e implantar a aplicação no AWS ECS.
2. Verifique as mensagens do script para acompanhar a criação ou atualização da Task Definition e do serviço ECS.
3. Obtenha o IP público da nova task e teste o endpoint da API.