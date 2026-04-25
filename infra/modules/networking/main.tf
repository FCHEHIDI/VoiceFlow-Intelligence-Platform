######################################################################
# VoiceFlow networking module
#
# Provisions:
#   * a /16 VPC with DNS support and hostnames enabled
#   * two public, two private and two DB subnets across two AZs
#   * an internet gateway and one or more NAT gateways
#   * route tables for public (igw) / private (nat) / db (no internet)
#   * five least-privilege security groups (alb / ml / inference / rds / redis)
#   * optional VPC Flow Logs to CloudWatch
######################################################################

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name_prefix = "voiceflow-${var.env}"
  azs         = slice(data.aws_availability_zones.available.names, 0, 2)
  nat_count   = var.enable_multi_az_nat ? 2 : 1
}

######################################################################
# VPC
######################################################################

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${local.name_prefix}-vpc"
  }
}

resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 1)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${local.name_prefix}-public-${count.index + 1}"
    Tier = "public"
  }
}

resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = local.azs[count.index]

  tags = {
    Name = "${local.name_prefix}-private-${count.index + 1}"
    Tier = "private"
  }
}

resource "aws_subnet" "db" {
  count = 2

  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 20)
  availability_zone = local.azs[count.index]

  tags = {
    Name = "${local.name_prefix}-db-${count.index + 1}"
    Tier = "db"
  }
}

######################################################################
# Internet + NAT
######################################################################

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${local.name_prefix}-igw"
  }
}

resource "aws_eip" "nat" {
  count = local.nat_count

  domain     = "vpc"
  depends_on = [aws_internet_gateway.this]

  tags = {
    Name = "${local.name_prefix}-nat-eip-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "this" {
  count = local.nat_count

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  depends_on    = [aws_internet_gateway.this]

  tags = {
    Name = "${local.name_prefix}-nat-${count.index + 1}"
  }
}

######################################################################
# Route tables
######################################################################

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = {
    Name = "${local.name_prefix}-rt-public"
  }
}

resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  count = 2

  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = var.enable_multi_az_nat ? aws_nat_gateway.this[count.index].id : aws_nat_gateway.this[0].id
  }

  tags = {
    Name = "${local.name_prefix}-rt-private-${count.index + 1}"
  }
}

resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# DB subnets are isolated; no default route to the internet.
resource "aws_route_table" "db" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${local.name_prefix}-rt-db"
  }
}

resource "aws_route_table_association" "db" {
  count = 2

  subnet_id      = aws_subnet.db[count.index].id
  route_table_id = aws_route_table.db.id
}

######################################################################
# Security groups
######################################################################

resource "aws_security_group" "alb" {
  name_prefix = "${local.name_prefix}-alb-"
  description = "VoiceFlow ${var.env} ALB: ingress 80/443 from internet"
  vpc_id      = aws_vpc.this.id

  ingress {
    description = "HTTP from internet (redirected to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS from internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${local.name_prefix}-alb-sg"
  }
}

resource "aws_security_group" "ml_service" {
  name_prefix = "${local.name_prefix}-ml-"
  description = "VoiceFlow ${var.env} ml-service: ingress 8000 from ALB only"
  vpc_id      = aws_vpc.this.id

  egress {
    description = "Allow all egress (ECR, Secrets Manager, RDS, Redis, Rust service)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${local.name_prefix}-ml-sg"
  }
}

resource "aws_security_group_rule" "ml_from_alb" {
  type                     = "ingress"
  from_port                = 8000
  to_port                  = 8000
  protocol                 = "tcp"
  security_group_id        = aws_security_group.ml_service.id
  source_security_group_id = aws_security_group.alb.id
  description              = "ml-service 8000 from ALB"
}

resource "aws_security_group" "inference" {
  name_prefix = "${local.name_prefix}-inf-"
  description = "VoiceFlow ${var.env} inference-engine: ingress 3000 from ALB and ml-service"
  vpc_id      = aws_vpc.this.id

  egress {
    description = "Allow all egress (ECR, Secrets Manager)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${local.name_prefix}-inf-sg"
  }
}

resource "aws_security_group_rule" "inf_from_alb" {
  type                     = "ingress"
  from_port                = 3000
  to_port                  = 3000
  protocol                 = "tcp"
  security_group_id        = aws_security_group.inference.id
  source_security_group_id = aws_security_group.alb.id
  description              = "inference 3000 from ALB"
}

resource "aws_security_group_rule" "inf_from_ml" {
  type                     = "ingress"
  from_port                = 3000
  to_port                  = 3000
  protocol                 = "tcp"
  security_group_id        = aws_security_group.inference.id
  source_security_group_id = aws_security_group.ml_service.id
  description              = "inference 3000 from ml-service (internal /infer calls)"
}

resource "aws_security_group" "rds" {
  name_prefix = "${local.name_prefix}-rds-"
  description = "VoiceFlow ${var.env} RDS: ingress 5432 from ml-service only"
  vpc_id      = aws_vpc.this.id

  egress {
    description = "Allow all egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${local.name_prefix}-rds-sg"
  }
}

resource "aws_security_group_rule" "rds_from_ml" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.rds.id
  source_security_group_id = aws_security_group.ml_service.id
  description              = "Postgres 5432 from ml-service"
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name_prefix}-redis-"
  description = "VoiceFlow ${var.env} Redis: ingress 6379 from ml-service only"
  vpc_id      = aws_vpc.this.id

  egress {
    description = "Allow all egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${local.name_prefix}-redis-sg"
  }
}

resource "aws_security_group_rule" "redis_from_ml" {
  type                     = "ingress"
  from_port                = 6379
  to_port                  = 6379
  protocol                 = "tcp"
  security_group_id        = aws_security_group.redis.id
  source_security_group_id = aws_security_group.ml_service.id
  description              = "Redis 6379 from ml-service"
}

######################################################################
# VPC Flow Logs (optional, on by default)
######################################################################

resource "aws_cloudwatch_log_group" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name              = "/aws/vpc/${local.name_prefix}/flow-logs"
  retention_in_days = var.flow_logs_retention_days
}

data "aws_iam_policy_document" "flow_logs_assume" {
  count = var.enable_flow_logs ? 1 : 0

  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["vpc-flow-logs.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "flow_logs_policy" {
  count = var.enable_flow_logs ? 1 : 0

  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name_prefix        = "${local.name_prefix}-flow-logs-"
  assume_role_policy = data.aws_iam_policy_document.flow_logs_assume[0].json
}

resource "aws_iam_role_policy" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name_prefix = "${local.name_prefix}-flow-logs-"
  role        = aws_iam_role.flow_logs[0].id
  policy      = data.aws_iam_policy_document.flow_logs_policy[0].json
}

resource "aws_flow_log" "this" {
  count = var.enable_flow_logs ? 1 : 0

  vpc_id          = aws_vpc.this.id
  iam_role_arn    = aws_iam_role.flow_logs[0].arn
  log_destination = aws_cloudwatch_log_group.flow_logs[0].arn
  traffic_type    = "ALL"

  tags = {
    Name = "${local.name_prefix}-flow-logs"
  }
}
