env              = "prod"
instance_type    = "t2.nano"
key_pair_name    = "makhoul-polyai-prod-key"
region           = "eu-central-1"
s3_bucket_prefix = "makhoul-polyai-prod-"

private_subnet_cidrs = [
  "10.0.1.0/24",
  "10.0.2.0/24",
]

public_subnet_cidrs = [
  "10.0.101.0/24",
  "10.0.102.0/24",
]

vpc_cidr = "10.0.0.0/16"
