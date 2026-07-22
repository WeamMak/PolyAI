variable "ami_id" {
  description = "EC2 Ubuntu AMI"
  type        = string
}

variable "availability_zone" {
  description = "Availability zone for the EC2 instance"
  type        = string
}

variable "env" {
  description = "Deployment environment"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "key_pair_name" {
  description = "Name of the EC2 key pair"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "s3_bucket_prefix" {
  description = "Prefix for the application S3 bucket name"
  type        = string
}
