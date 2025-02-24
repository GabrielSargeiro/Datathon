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

---

## Build, Push e Deploy no AWS

O script `build_push_aws.sh` executa todo o fluxo de automação.

### Passos para executar:

1. **Abra o Docker Desktop** e, em seguida, abra o Git Bash na pasta raiz do projeto.

2. Execute o script:

   ```bash
   ./build_push_aws.sh
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