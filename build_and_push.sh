#!/bin/bash
set -e

echo "======================================"
echo "Carregando variáveis do arquivo .env..."
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Arquivo .env não encontrado. Verifique se ele existe na raiz do projeto."
  exit 1
fi

# Variáveis obrigatórias para ECR e ECS
: "${AWS_REGION:?Variável AWS_REGION não definida no .env}"
: "${AWS_ACCOUNT_ID:?Variável AWS_ACCOUNT_ID não definida no .env}"
: "${AWS_ACCESS_KEY_ID:?Variável AWS_ACCESS_KEY_ID não definida no .env}"
: "${AWS_SECRET_ACCESS_KEY:?Variável AWS_SECRET_ACCESS_KEY não definida no .env}"
: "${AWS_SESSION_TOKEN:?Variável AWS_SESSION_TOKEN não definida no .env}"
: "${REPO_NAME:?Variável REPO_NAME não definida no .env}"
: "${IMAGE_TAG:=latest}"
: "${ECS_CLUSTER:?Variável ECS_CLUSTER não definida no .env}"
: "${ECS_SERVICE:?Variável ECS_SERVICE não definida no .env}"
: "${SUBNETS:?Variável SUBNETS não definida no .env}"
: "${SECURITY_GROUPS:?Variável SECURITY_GROUPS não definida no .env}"

echo "======================================"
echo "AWS_REGION: $AWS_REGION"
echo "AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
echo "REPO_NAME: $REPO_NAME"
echo "IMAGE_TAG: $IMAGE_TAG"
echo "ECS_CLUSTER: $ECS_CLUSTER"
echo "ECS_SERVICE: $ECS_SERVICE"
echo "======================================"

### Parte 1: Build e Push da Imagem para o ECR

echo "[1/5] Realizando login no AWS ECR..."
"/c/Program Files/Amazon/AWSCLIV2/aws" ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "[2/5] Verificando se o repositório $REPO_NAME existe no ECR..."
if "/c/Program Files/Amazon/AWSCLIV2/aws" ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Repositório $REPO_NAME já existe."
else
    echo "Repositório $REPO_NAME não encontrado. Criando repositório..."
    "/c/Program Files/Amazon/AWSCLIV2/aws" ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION"
fi

echo "[3/5] Construindo a imagem Docker..."
docker build -t "$REPO_NAME" -f docker/Dockerfile .

echo "[4/5] Tagueando a imagem..."
docker tag "$REPO_NAME:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"

echo "[5/5] Enviando a imagem para o ECR..."
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
echo "Imagem enviada com sucesso!"
echo "======================================"

### Parte 2: Deploy no ECS

echo "Verificando se o cluster $ECS_CLUSTER existe e está ativo..."
CLUSTER_STATUS=$(aws ecs describe-clusters --clusters "$ECS_CLUSTER" --query "clusters[0].status" --output text)
if [ "$CLUSTER_STATUS" != "ACTIVE" ]; then
  echo "Cluster $ECS_CLUSTER não existe ou não está ativo. Criando cluster..."
  aws ecs create-cluster --cluster-name "$ECS_CLUSTER"
else
  echo "Cluster $ECS_CLUSTER está ativo."
fi
echo "======================================"

# Define a nova imagem completa e o nome da task
NEW_IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
TASK_FAMILY="datathon-task"

echo "Gerando arquivo task_definition.json com a imagem $NEW_IMAGE..."
TASK_DEF_FILE="task_definition.json"
cat <<EOF > $TASK_DEF_FILE
{
    "family": "$TASK_FAMILY",
    "containerDefinitions": [
        {
            "name": "api",
            "image": "$NEW_IMAGE",
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
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                },
                "secretOptions": []
            },
            "systemControls": []
        }
    ],
    "taskRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/LabRole",
    "executionRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/LabRole",
    "networkMode": "awsvpc",
    "volumes": [],
    "placementConstraints": [],
    "requiresCompatibilities": [
        "FARGATE"
    ],
    "cpu": "2048",
    "memory": "14336"
}
EOF
echo "Arquivo task_definition.json gerado."
echo "======================================"

echo "Registrando nova revisão da Task Definition..."
NEW_TASK_DEF_ARN=$(aws ecs register-task-definition --cli-input-json file://$TASK_DEF_FILE --query "taskDefinition.taskDefinitionArn" --output text)
echo "Nova Task Definition registrada: $NEW_TASK_DEF_ARN"
echo "======================================"

# Formata as listas de subnets e security groups para o JSON
SUBNETS_JSON=$(echo $SUBNETS | sed 's/,/","/g' | sed 's/^/"/' | sed 's/$/"/')
SECURITY_GROUPS_JSON=$(echo $SECURITY_GROUPS | sed 's/,/","/g' | sed 's/^/"/' | sed 's/$/"/')

echo "Verificando se o serviço $ECS_SERVICE existe no cluster $ECS_CLUSTER..."
SERVICE_CHECK=$(aws ecs describe-services --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" 2>&1 || true)

if echo "$SERVICE_CHECK" | grep -q "MISSING"; then
    echo "Serviço $ECS_SERVICE não encontrado. Iremos criá-lo..."
    aws ecs create-service \
      --cluster "$ECS_CLUSTER" \
      --service-name "$ECS_SERVICE" \
      --task-definition "$NEW_TASK_DEF_ARN" \
      --desired-count 1 \
      --launch-type FARGATE \
      --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS_JSON],securityGroups=[$SECURITY_GROUPS_JSON],assignPublicIp=ENABLED}"
    echo "Serviço $ECS_SERVICE criado com sucesso!"
else
    echo "Serviço $ECS_SERVICE encontrado. Tentando atualizar para a nova Task Definition..."
    if ! aws ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" --task-definition "$NEW_TASK_DEF_ARN" --force-new-deployment; then
        echo "Erro ao atualizar serviço (ServiceNotActiveException provável)."
        echo "Excluindo o serviço $ECS_SERVICE..."
        aws ecs delete-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" --force
        echo "Aguardando 30 segundos para a exclusão do serviço..."
        sleep 30
        echo "Recriando o serviço $ECS_SERVICE..."
        aws ecs create-service \
          --cluster "$ECS_CLUSTER" \
          --service-name "$ECS_SERVICE" \
          --task-definition "$NEW_TASK_DEF_ARN" \
          --desired-count 1 \
          --launch-type FARGATE \
          --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS_JSON],securityGroups=[$SECURITY_GROUPS_JSON],assignPublicIp=ENABLED}"
        echo "Serviço $ECS_SERVICE recriado com sucesso!"
    else
        echo "Serviço $ECS_SERVICE atualizado com sucesso para a nova Task Definition."
    fi
fi
echo "======================================"

echo "Listando todas as tasks do serviço $ECS_SERVICE..."
TASK_ARNS=$(aws ecs list-tasks --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" --query 'taskArns[]' --output text)
if [ -n "$TASK_ARNS" ]; then
  echo "Forçando parada das tasks antigas..."
  for TASK in $TASK_ARNS; do
    echo "Parando task: $TASK"
    aws ecs stop-task --cluster "$ECS_CLUSTER" --task "$TASK" --reason "Deploy update: stopping old task"
  done
else
  echo "Nenhuma task em execução encontrada."
fi
echo "======================================"

echo "Aguardando 30 segundos para a nova task iniciar..."
sleep 30

echo "Obtendo o ARN da primeira task em execução..."
TASK_ARN=$(aws ecs list-tasks --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" --query 'taskArns[0]' --output text)
if [ "$TASK_ARN" = "None" ]; then
  echo "Nenhuma task encontrada em execução."
  exit 1
fi
echo "Task ARN: $TASK_ARN"
echo "======================================"

echo "Obtendo o ID da ENI da task..."
ENI_ID=$(aws ecs describe-tasks --cluster "$ECS_CLUSTER" --tasks "$TASK_ARN" \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
if [ -z "$ENI_ID" ]; then
  echo "Não foi possível obter o ID da interface de rede."
  exit 1
fi
echo "ENI ID: $ENI_ID"
echo "======================================"

echo "Consultando o IP público associado à ENI..."
PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI_ID" \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
if [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" = "None" ]; then
  echo "Não foi possível obter o IP público da ENI."
else
  echo "IP público: $PUBLIC_IP "
  echo "Healthcheck da API: http://$PUBLIC_IP:5000/healthcheck"
fi

echo "======================================"
echo "Processo de build, push e deploy concluído!"
echo "======================================"

# Evita a conversão de caminhos no Git Bash
export MSYS_NO_PATHCONV=1

LOG_GROUP="/ecs/datathon-task"

echo "Iniciando busca do log 'Application startup complete' a cada 20 sec"
MAX_ATTEMPTS=30
attempt=1

while [ $attempt -le $MAX_ATTEMPTS ]; do
  START_TIME=$(($(date +%s) - 300))
  LOG_OUTPUT=$(aws logs filter-log-events \
    --log-group-name "$LOG_GROUP" \
    --filter-pattern "Application startup complete" \
    --start-time $((START_TIME * 1000)) \
    --output text)

  if echo "$LOG_OUTPUT" | grep -q "Application startup complete"; then
      echo "Log encontrado: Application startup complete"
      break
  else
      echo "Tentativa $attempt: Aguardando Application startup..."
      sleep 20
      attempt=$((attempt+1))
  fi
done

if [ $attempt -gt $MAX_ATTEMPTS ]; then
    echo "Log 'Application startup complete' não encontrado após $(($MAX_ATTEMPTS * 20)) segundos."
    exit 1
fi