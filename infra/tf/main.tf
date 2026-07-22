terraform {
  required_version = ">= 1.7.0"

  backend "s3" {
    bucket       = "weam-polyai-tfstate-us-east-1"
    key          = "polyai/dev/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.55"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = "default"
}

module "polyai_service_vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.8.1"

  name = "makhoul-polyai-${var.env}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway = false

  tags = {
    Env = var.env
  }
}

resource "aws_security_group" "polyai_dev_sg" {
  name        = "makhoul-polyai-${var.env}-sg"
  description = "Allow SSH and HTTP traffic"
  vpc_id      = module.polyai_service_vpc.vpc_id

  ingress {
    description = "Allow SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Allow HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "makhoul-polyai-${var.env}-sg"
    Env  = var.env
  }
}

resource "aws_key_pair" "polyai_dev" {
  key_name   = var.key_pair_name
  public_key = file(pathexpand("~/.ssh/polyai_dev.pub"))

  tags = {
    Name = var.key_pair_name
    Env  = var.env
  }
}

resource "aws_s3_bucket" "polyai_dev" {
  bucket_prefix = var.s3_bucket_prefix

  tags = {
    Name = "makhoul-polyai-${var.env}"
    Env  = var.env
  }
}

resource "aws_instance" "polyai_dev" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  subnet_id                   = module.polyai_service_vpc.public_subnets[0]
  associate_public_ip_address = true

  key_name               = aws_key_pair.polyai_dev.key_name
  vpc_security_group_ids = [aws_security_group.polyai_dev_sg.id]

  # The application is assumed to require the bucket during startup.
  depends_on = [aws_s3_bucket.polyai_dev]

  tags = {
    Name      = "makhoul-polyai-${var.env}"
    Env       = var.env
    CreatedBy = "makhoul"
    Author    = "Makhoul"
  }
}

resource "aws_ebs_volume" "polyai_dev_data" {
  availability_zone = aws_instance.polyai_dev.availability_zone
  size              = 5
  type              = "gp3"
  encrypted         = true

  tags = {
    Name = "makhoul-polyai-${var.env}-data"
    Env  = var.env
  }
}

resource "aws_volume_attachment" "polyai_dev_data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.polyai_dev_data.id
  instance_id = aws_instance.polyai_dev.id
}

