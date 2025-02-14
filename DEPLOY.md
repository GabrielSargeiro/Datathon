# Documentação Deploy - Datathon System
 
Este documento descreve o fluxo completo para construir, enviar e implantar a aplicação no AWS ECS (Fargate).
 
## Requisitos do Ambiente Local
 
Certifique-se de que sua máquina possua os seguintes itens instalados e configurados:
- **Docker Desktop:** Necessário para construir e executar imagens Docker.
- **Git Bash (ou outro terminal Unix-like):** Recomendado para executar scripts Bash.
- **Credenciais AWS configuradas:** Um arquivo `.env` com suas credenciais.

## Sumário

1. [Build e Push da Imagem Docker](#build-e-push-da-imagem-docker)
2. [Criação da Task Definition](#criação-da-task-definition)
3. [Testando a API](#testando-a-api)
4. [Conclusão](#conclusão)

---

## Build e Push da Imagem Docker

A imagem Docker é construída e enviada para o AWS ECR utilizando o script `build_and_push.sh`. Certifique-se de ter um arquivo `.env` na mesma pasta configurado com as seguintes variáveis (ajuste os valores):

```dotenv
AWS_REGION=
AWS_ACCOUNT_ID=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SESSION_TOKEN=
REPO_NAME=datathon-recommendation
IMAGE_TAG=latest
```

### Passos para executar:

1. Abra o Docker e abra o Git Bash na pasta raiz do projeto

2. **Torne o script executável (se necessário):**
   ```bash
   chmod +x build_and_push.sh
   ```
3. **Execute o script:**
   ```bash
   ./build_and_push.sh
   ```

O script realiza as seguintes ações:
- Carrega as variáveis do arquivo `.env`.
- Faz login no AWS ECR.
- Verifica se o repositório existe e o cria, se necessário.
- Constrói a imagem Docker utilizando o Dockerfile localizado na pasta `docker/`.
- Tagueia e envia (push) a imagem para o ECR.

---

## Criação da Task Definition

Para criar a Task Definition no Amazon ECS, utilize o seguinte JSON. Este JSON define a task chamada "datathon-task" com os seguintes parâmetros:

- **Família:** `datathon-task`
- **Tipo de Inicialização:** Fargate (compatibilidade: `"FARGATE"`)
- **Modo de Rede:** `awsvpc`
- **Recursos:** 4 vCPUs (4096 unidades) e 8 GB de memória (8192 MB)
- **Container "api":** Utiliza a imagem `140867038646.dkr.ecr.us-east-1.amazonaws.com/datathon-recommendation:latest` e mapeia a porta 5000 do container para a porta 5000 do host.
- **Papéis:** Usa o IAM Role `LabRole` para `taskRoleArn` e `executionRoleArn`
- **Logs:** Configurado para enviar logs para o CloudWatch (grupo `/ecs/datathon-task`, região `us-east-1`, stream prefix `ecs`)

Para criar a Task Definition, copie o JSON abaixo e registre-o via AWS CLI ou pelo Console do ECS:

```json
{
    "family": "datathon-task",
    "containerDefinitions": [
        {
            "name": "api",
            "image": "140867038646.dkr.ecr.us-east-1.amazonaws.com/datathon-recommendation:latest",
            "cpu": 0,
            "portMappings": [
                {
                    "name": "api-5000-tcp",
                    "containerPort": 5000,
                    "hostPort": 5000,
                    "protocol": "tcp"
                }
            ],
            "essential": true,
            "environment": [],
            "mountPoints": [],
            "volumesFrom": [],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/datathon-task",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                },
                "secretOptions": []
            },
            "systemControls": []
        }
    ],
    "taskRoleArn": "arn:aws:iam::140867038646:role/LabRole",
    "executionRoleArn": "arn:aws:iam::140867038646:role/LabRole",
    "networkMode": "awsvpc",
    "volumes": [],
    "placementConstraints": [],
    "requiresCompatibilities": [
        "FARGATE"
    ],
    "cpu": "4096",
    "memory": "8192"
}
```

## Testando a API

Após a atualização do serviço, aguarde alguns minutos até que a nova task esteja em execução.

1. **Obtenha o IP Público da Task:**
   - No Console do ECS, clique na task em execução e, na seção **Networking**, verifique o **Public IP** atribuído (por exemplo, `44.215.126.227`).

2. **Teste o Endpoint:**
   Use `curl` ou Postman para enviar uma requisição:
   ```bash
   curl.exe -v -X POST http://44.215.126.227:5000/recommend -H "Content-Type: application/json" -d '{"userId": "seu_userId_valido"}'
   ```
   
---

## Conclusão

1. **Execute o script `build_and_push.sh`** para construir e enviar a nova imagem Docker para o ECR.
2. **Registre uma nova revisão da Task Definition** para usar a nova imagem.
3. **Crie o serviço ECS** e implante ele parar poder acessar.
4. **Teste o endpoint** da API usando Postman ou curl utilizando o IP publico.

