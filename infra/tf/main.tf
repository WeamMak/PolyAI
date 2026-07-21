terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.55"
    }
  }

  required_version = ">= 1.7.0"
}

provider "aws" {
  region  = "us-east-1"
  profile = "default"
}

resource "aws_security_group" "polyai_dev_sg" {
  name        = "makhoul-polyai-dev-sg"
  description = "Allow SSH and HTTP traffic"

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
    Name = "makhoul-polyai-dev-sg"
    Env  = "dev"
  }
}

resource "aws_key_pair" "polyai_dev" {
  key_name   = "makhoul-polyai-dev-key"
  public_key = file(pathexpand("~/.ssh/polyai_dev.pub"))

  tags = {
    Name = "makhoul-polyai-dev-key"
    Env  = "dev"
  }
}

resource "aws_s3_bucket" "polyai_dev" {
  bucket_prefix = "makhoul-polyai-dev-"

  tags = {
    Name = "makhoul-polyai-dev"
    Env  = "dev"
  }
}

resource "aws_instance" "polyai_dev" {
  ami           = "ami-0b6d9d3d33ba97d99"
  instance_type = "t2.nano"

  key_name               = aws_key_pair.polyai_dev.key_name
  vpc_security_group_ids = [aws_security_group.polyai_dev_sg.id]

  # The application is assumed to require the bucket during startup.
  depends_on = [aws_s3_bucket.polyai_dev]

  tags = {
    Name      = "makhoul-polyai-dev"
    Env       = "dev"
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
    Name = "makhoul-polyai-dev-data"
    Env  = "dev"
  }
}

resource "aws_volume_attachment" "polyai_dev_data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.polyai_dev_data.id
  instance_id = aws_instance.polyai_dev.id
}