env              = "dev"
instance_type    = "t2.nano"
key_pair_name    = "makhoul-polyai-dev-key"
region           = "us-east-1"
s3_bucket_prefix = "makhoul-polyai-dev-"

private_subnet_cidrs = [
  "10.0.1.0/24",
  "10.0.2.0/24",
]

public_subnet_cidrs = [
  "10.0.101.0/24",
  "10.0.102.0/24",
]

vpc_cidr = "10.0.0.0/16"
