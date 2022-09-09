terraform {
  required_version = "1.2.2"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "= 4.16.0"
    }
  }
  backend "s3" {
    bucket         = "hydra-boosters-terraform"
    key            = "main.state"
    dynamodb_table = "terraform"
  }
}

provider "aws" {
  region = "us-east-2"
}

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# resources that were created manually
data "aws_secretsmanager_secret" "grafana-push-secret" {
  arn = "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:grafana-push-l8vSeT"
}
data "aws_secretsmanager_secret" "docker-pull-secret" {
  arn = "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:docker-pull-8ysWH3"
}
# generated with: openssl rand -base64 32
data "aws_secretsmanager_secret" "hydra-random-seed" {
  arn = "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:hydra-random-seed-nNh40D"
}
# seed for the test flight, so that they have different but consistent peer IDs
data "aws_secretsmanager_secret" "hydra-random-seed-test" {
  arn = "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:hydra-random-seed-test-lxogsg"
}
# from https://github.com/protocol/monitoring-infra/blob/master/ansible/vault.yml
data "aws_secretsmanager_secret" "push-gateway-basicauth" {
  arn = "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:push-gateway-basicauth-remzOi"
}
data "aws_kms_key" "default_secretsmanager_key" {
  key_id = "arn:aws:kms:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:key/58e59216-a463-4b1c-917c-8ed676da68b0"
}
data "aws_vpc" "main_vpc" {
  id = "vpc-0a5b217ef1e275e39"
}
data "aws_subnet" "subnet_use2-az1" {
  id = "subnet-0a38e86cd46c1e109"
}
data "aws_subnet" "subnet_use2-az2" {
  id = "subnet-0c9532d516223c525"
}
module "ecs" {
  source             = "terraform-aws-modules/ecs/aws"
  version            = "3.5.0"
  name               = var.name
  container_insights = true
  capacity_providers = ["FARGATE"]

  default_capacity_provider_strategy = [
    {
      capacity_provider = "FARGATE"
      weight            = "1"
    }
  ]
}

resource "aws_dynamodb_table" "main" {
  name           = "${var.name}-table"
  billing_mode   = "PROVISIONED"
  read_capacity  = 4000
  write_capacity = 22000
  hash_key       = "key"
  range_key      = "ttl"

  point_in_time_recovery {
    enabled = true
  }

  attribute {
    name = "key"
    type = "B"
  }

  attribute {
    name = "ttl"
    type = "N"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  # ignore changes to read and write capacity, as they are managed by Application Auto Scaling
  lifecycle {
    ignore_changes = [read_capacity, write_capacity]
  }
}

resource "aws_appautoscaling_target" "providers_read_target" {
  max_capacity       = 30000
  min_capacity       = 1000
  resource_id        = "table/${aws_dynamodb_table.main.name}"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}

resource "aws_appautoscaling_policy" "providers_read_policy" {
  name               = "DynamoDBReadCapacityUtilization:${aws_appautoscaling_target.providers_read_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.providers_read_target.resource_id
  scalable_dimension = aws_appautoscaling_target.providers_read_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.providers_read_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBReadCapacityUtilization"
    }
    target_value       = 90
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
}

resource "aws_appautoscaling_target" "providers_write_target" {
  max_capacity       = 60000
  min_capacity       = 1000
  resource_id        = "table/${aws_dynamodb_table.main.name}"
  scalable_dimension = "dynamodb:table:WriteCapacityUnits"
  service_namespace  = "dynamodb"
}

resource "aws_appautoscaling_policy" "providers_write_policy" {
  name               = "DynamoDBWriteCapacityUtilization:${aws_appautoscaling_target.providers_write_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.providers_write_target.resource_id
  scalable_dimension = aws_appautoscaling_target.providers_write_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.providers_write_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBWriteCapacityUtilization"
    }
    target_value       = 90
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
}

resource "aws_dynamodb_table" "ipns" {
  name           = "${var.name}-ipns-table"
  billing_mode   = "PROVISIONED"
  read_capacity  = 100
  write_capacity = 100
  hash_key       = "DSKey"

  point_in_time_recovery {
    enabled = true
  }

  attribute {
    name = "DSKey"
    type = "S"
  }

  # ignore changes to read and write capacity, as they are managed by Application Auto Scaling
  lifecycle {
    ignore_changes = [read_capacity, write_capacity]
  }
}

resource "aws_appautoscaling_target" "ipns_read_target" {
  max_capacity       = 10000
  min_capacity       = 100
  resource_id        = "table/${aws_dynamodb_table.ipns.name}"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}

resource "aws_appautoscaling_policy" "ipns_read_policy" {
  name               = "DynamoDBReadCapacityUtilization:${aws_appautoscaling_target.ipns_read_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ipns_read_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ipns_read_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ipns_read_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBReadCapacityUtilization"
    }
    target_value       = 90
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
}

resource "aws_appautoscaling_target" "ipns_write_target" {
  max_capacity       = 10000
  min_capacity       = 100
  resource_id        = "table/${aws_dynamodb_table.ipns.name}"
  scalable_dimension = "dynamodb:table:WriteCapacityUnits"
  service_namespace  = "dynamodb"
}

resource "aws_appautoscaling_policy" "ipns_write_policy" {
  name               = "DynamoDBWriteCapacityUtilization:${aws_appautoscaling_target.ipns_write_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ipns_write_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ipns_write_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ipns_write_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBWriteCapacityUtilization"
    }
    target_value       = 90
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
}

resource "aws_vpc_endpoint" "s3_vpce" {
  vpc_id          = data.aws_vpc.main_vpc.id
  service_name    = "com.amazonaws.${data.aws_region.current.name}.s3"
  route_table_ids = [data.aws_vpc.main_vpc.main_route_table_id]
}
resource "aws_s3_bucket" "grafana_config_bucket" {
  bucket = "grafana-config-${data.aws_caller_identity.current.account_id}-${data.aws_region.current.name}"
}
# this allows unauthenticated access only via the VPCe (so only within the VPC)
resource "aws_s3_bucket_policy" "grafana_config_bucket_policy" {
  bucket = aws_s3_bucket.grafana_config_bucket.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = ""
        Principal = "*"
        Action    = "s3:GetObject"
        Effect    = "Allow"
        Resource  = ["${aws_s3_bucket.grafana_config_bucket.arn}/*"]
        Condition = {
          StringEquals = {
            "aws:sourceVpce" = aws_vpc_endpoint.s3_vpce.id
          }
        }
      }
    ]
  })
}

resource "aws_s3_bucket_object" "grafana_agent_config_object" {
  # we need to re-deploy if this changes, so we need to change the key every time the file changes
  # which will force an ECS redeployment
  key    = filemd5("grafana-agent-config.yaml")
  bucket = aws_s3_bucket.grafana_config_bucket.id
  source = "grafana-agent-config.yaml"
}

resource "aws_ecs_service" "test-hydra" {
  count                  = var.hydra_count
  name                   = "hydra-booster-${var.name}-${count.index}"
  cluster                = module.ecs.ecs_cluster_id
  task_definition        = aws_ecs_task_definition.hydra-booster[count.index].arn
  desired_count          = 1
  enable_execute_command = true

  network_configuration {
    subnets          = [data.aws_subnet.subnet_use2-az1.id]
    security_groups  = [aws_security_group.hydra.id]
    assign_public_ip = true
  }

  capacity_provider_strategy {
    base              = 0
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

resource "aws_iam_role" "main" {
  name = var.name
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy" "main" {
  name = var.name
  role = aws_iam_role.main.id

  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "HydraDynamoDB",
        "Effect" : "Allow",
        "Action" : [
          "dynamodb:PutItem",
          "dynamodb:Query",
          "dynamodb:DescribeTable"
        ],
        "Resource" : aws_dynamodb_table.main.arn
      },
      {
        "Sid" : "HydraIPNSDynamoDB",
        "Effect" : "Allow",
        "Action" : [
          "dynamodb:*",
        ],
        "Resource" : aws_dynamodb_table.ipns.arn
      },
      # allow SSH access via SSM
      {
        "Effect" : "Allow",
        "Action" : [
          "ssmmessages:CreateControlChannel",
          "ssmmessages:CreateDataChannel",
          "ssmmessages:OpenControlChannel",
          "ssmmessages:OpenDataChannel"
        ],
        "Resource" : "*"
      }
    ]
  })
}

# note: to keep terraform from recreating this every time, keep the container definition JSON alphabetized
resource "aws_ecs_task_definition" "hydra-booster" {
  count                    = var.hydra_count
  family                   = "hydra-booster-${var.name}-${count.index}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  execution_role_arn       = aws_iam_role.ecsTaskExecutionRole.arn
  task_role_arn            = aws_iam_role.main.arn
  depends_on               = [aws_iam_role.main]

  cpu      = 4 * 1024  # max is 4*1024
  memory   = 20 * 1024 # max is 30*1024
  tags     = {}
  tags_all = {}
  container_definitions = jsonencode([
    {
      cpu   = 0
      image = "libp2p/hydra-booster:992a8ef"
      environment = [
        { name = "HYDRA_NHEADS", value = tostring(var.hydra_nheads) },
        { name = "HYDRA_NAME", value = "${var.name}-${count.index}" },
        { name = "HYDRA_BOOTSTRAP_PEERS", value = "/dnsaddr/sjc-2.bootstrap.libp2p.io/p2p/QmZa1sAxajnQjVM8WjWXoMbmPd7NsWhfKsPkErzpm9wGkp,/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN,/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa,/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb,/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt" },
        { name = "HYDRA_DISABLE_PREFETCH", value = "false" },
        { name = "HYDRA_PORT_BEGIN", value = "30000" },
        { name = "HYDRA_ID_OFFSET", value = tostring(count.index * var.hydra_nheads) },
        { name = "HYDRA_PROVIDER_STORE", value = "dynamodb://table=${aws_dynamodb_table.main.name},ttl=24h,queryLimit=10000" },
        { name = "HYDRA_STORE_THE_INDEX_ADDR", value = "https://infra.cid.contact/multihash" },
        { name = "HYDRA_DELEGATED_ROUTING_TIMEOUT", value = "1000" },
        { name = "HYDRA_DB", value = "dynamodb://table=${aws_dynamodb_table.ipns.name}" },
        { name = "HYDRA_PSTORE", value = "pstore"}
      ]
      essential = true
      healthCheck = {
        # if a host is totally dead, we want to replace it
        # but if it's just really busy, we generally want to leave it alone
        # so these health checks are pretty liberal with lots of retries
        command     = ["CMD-SHELL", "curl -fsS -o /dev/null localhost:8888/metrics || exit 1"],
        interval    = 30, # seconds
        retries     = 10,
        startPeriod = 300, # seconds
        timeout     = 10   # seconds
      }
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.logs.name,
          awslogs-region        = "${data.aws_region.current.name}",
          awslogs-stream-prefix = "ecs"
        }
      }
      mountPoints  = []
      name         = "hydra"
      portMappings = [{ containerPort = 8888, hostPort = 8888, protocol = "tcp" }]
      secrets = [
        { name = "HYDRA_RANDOM_SEED", valueFrom = "${data.aws_secretsmanager_secret.hydra-random-seed.arn}:seed::" }
      ]
      ulimits = [
        {
          name      = "nofile",
          hardLimit = 1048576,
          softLimit = 1048576
        }
      ]
      volumesFrom = []
    },
    {
      command = [
        "--prometheus.wal-directory=/etc/agent/data",
        "--enable-features=remote-configs",
        "--config.expand-env",
        "--config.file=https://${aws_s3_bucket.grafana_config_bucket.bucket_regional_domain_name}/${aws_s3_bucket_object.grafana_agent_config_object.id}"
      ]
      cpu   = 0
      image = "grafana/agent:v0.23.0"
      environment = [
        # we use this for setting labels on metrics
        { name = "HYDRA_NAME", value = "${var.name}-${count.index}" }
      ]
      essential = true
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.logs.name,
          awslogs-region        = "${data.aws_region.current.name}",
          awslogs-stream-prefix = "ecs"
        }
      }
      mountPoints  = []
      name         = "grafana-agent"
      portMappings = []
      repositoryCredentials = {
        credentialsParameter = data.aws_secretsmanager_secret.docker-pull-secret.arn
      }
      secrets = [
        { name = "GRAFANA_USER", valueFrom = "${data.aws_secretsmanager_secret.grafana-push-secret.arn}:username::" },
        { name = "GRAFANA_PASS", valueFrom = "${data.aws_secretsmanager_secret.grafana-push-secret.arn}:password::" }
      ]
      volumesFrom = []
    },
    {
      command = []
      cpu     = 0
      image   = "mcamou/docker-alpine-cron",
      environment = [
        {
          name  = "CRON_STRINGS",
          value = "* * * * * curl -s localhost:8888/metrics | curl --basic --user $${PUSHGATEWAY_USER}:$${PUSHGATEWAY_PASSWORD} --data-binary @- https://pushgateway.k8s.locotorp.info/metrics/job/hydra_boosters/instance/$${HYDRA_NAME}"
        },
        { name = "CRON_TAIL", value = "no_logfile" },
        { name = "CRON_CMD_OUTPUT_LOG", value = "1" },
        # we use this for setting labels on metrics
        { name = "HYDRA_NAME", value = "${var.name}-${count.index}" }
      ]
      essential = true
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.logs.name,
          awslogs-region        = "${data.aws_region.current.name}",
          awslogs-stream-prefix = "ecs"
        }
      }
      mountPoints  = []
      name         = "metrics-pushgateway"
      portMappings = []
      secrets = [
        { name = "PUSHGATEWAY_USER", valueFrom = "${data.aws_secretsmanager_secret.push-gateway-basicauth.arn}:user::" },
        { name = "PUSHGATEWAY_PASSWORD", valueFrom = "${data.aws_secretsmanager_secret.push-gateway-basicauth.arn}:password::" }
      ]
      volumesFrom = []
    }
  ])
}

resource "aws_security_group" "hydra" {
  name   = "hydra"
  vpc_id = "vpc-0a5b217ef1e275e39"

  ingress {
    description = "hydra-tcp"
    from_port   = 30000
    to_port     = 31000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    # ipv6_cidr_blocks = [aws_vpc.main.ipv6_cidr_block]
  }

  ingress {
    description = "hydra-udp"
    from_port   = 30000
    to_port     = 31000
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
    # ipv6_cidr_blocks = [aws_vpc.main.ipv6_cidr_block]
  }

  ingress {
    description = ""
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # should be just the lb
    # ipv6_cidr_blocks = [aws_vpc.main.ipv6_cidr_block]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "hydra"
  }
}

resource "aws_iam_role" "ecsTaskExecutionRole" {
  name               = "ecsTaskExecutionRole"
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
}

data "aws_iam_policy_document" "assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "ecsTaskExecutionRole_default_policy" {
  role       = aws_iam_role.ecsTaskExecutionRole.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "ecsTaskExecutionRole_policy_data" {
  statement {
    actions = ["kms:Decrypt", "secretsmanager:GetSecretValue"]
    resources = [
      data.aws_secretsmanager_secret.docker-pull-secret.arn,
      data.aws_secretsmanager_secret.grafana-push-secret.arn,
      data.aws_secretsmanager_secret.hydra-random-seed.arn,
      data.aws_secretsmanager_secret.hydra-random-seed-test.arn,
      data.aws_secretsmanager_secret.push-gateway-basicauth.arn,
      data.aws_kms_key.default_secretsmanager_key.arn,
    ]
  }
}

resource "aws_iam_policy" "ecsTaskExecutionRole_policy" {
  policy = data.aws_iam_policy_document.ecsTaskExecutionRole_policy_data.json
}

resource "aws_iam_role_policy_attachment" "ecsTaskExecutionRole_policy" {
  role       = aws_iam_role.ecsTaskExecutionRole.name
  policy_arn = aws_iam_policy.ecsTaskExecutionRole_policy.arn
}

resource "aws_cloudwatch_log_group" "logs" {
  name = var.name
}

# This is a one-host flight for testing changes before deploying them to the main flight.
# You should deploy to this first and validate that everything works, before deploying to the main flight.
module "test" {
  source                  = "./modules/hydra-flight"
  name                    = "test"
  hydra_count             = 1
  hydra_image             = "libp2p/hydra-booster:a6826f7"
  ecs_cluster_id          = module.ecs.ecs_cluster_id
  vpc_subnets             = [data.aws_subnet.subnet_use2-az1.id]
  security_groups         = [aws_security_group.hydra.id]
  execution_role_arn      = aws_iam_role.ecsTaskExecutionRole.arn
  task_role_arn           = aws_iam_role.main.arn
  docker_pull_secret_arn  = data.aws_secretsmanager_secret.docker-pull-secret.arn
  log_group_name          = aws_cloudwatch_log_group.logs.name
  grafana_config_endpoint = "https://${aws_s3_bucket.grafana_config_bucket.bucket_regional_domain_name}/${aws_s3_bucket_object.grafana_agent_config_object.id}"
  grafana_secrets = [
    { name = "GRAFANA_USER", valueFrom = "${data.aws_secretsmanager_secret.grafana-push-secret.arn}:username::" },
    { name = "GRAFANA_PASS", valueFrom = "${data.aws_secretsmanager_secret.grafana-push-secret.arn}:password::" }
  ]
  hydra_secrets = [
    { name = "HYDRA_RANDOM_SEED", valueFrom = "${data.aws_secretsmanager_secret.hydra-random-seed-test.arn}:seed::" }
  ]
  hydra_environment = [
    # Defaults 
    { name = "HYDRA_NHEADS", value = tostring(var.hydra_nheads) },
    { name = "HYDRA_BOOTSTRAP_PEERS", value = "/dnsaddr/sjc-2.bootstrap.libp2p.io/p2p/QmZa1sAxajnQjVM8WjWXoMbmPd7NsWhfKsPkErzpm9wGkp,/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN,/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa,/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb,/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt" },
    { name = "HYDRA_DISABLE_PREFETCH", value = "false" },
    { name = "HYDRA_PORT_BEGIN", value = "30000" },
    { name = "HYDRA_PROVIDER_STORE", value = "dynamodb://table=${aws_dynamodb_table.main.name},ttl=24h,queryLimit=10000" },
    { name = "HYDRA_REFRAME_ADDR", value = "http://cid.contact/reframe" },
    { name = "HYDRA_DELEGATED_ROUTING_TIMEOUT", value = "1000" },
    { name = "HYDRA_DB", value = "dynamodb://table=${aws_dynamodb_table.ipns.name}" }
  ]
}
