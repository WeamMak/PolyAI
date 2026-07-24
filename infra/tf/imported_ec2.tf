/* resource "aws_instance" "my_ec2" {
  ami                                  = "ami-0b6d9d3d33ba97d99"
  associate_public_ip_address          = true
  availability_zone                    = "us-east-1b"
  ebs_optimized                        = true
  iam_instance_profile                 = "EC2BedrockInvokeRole-weam-dev"
  instance_initiated_shutdown_behavior = "stop"
  instance_type                        = "t3.small"
  key_name                             = "weam-key1"
  placement_partition_number           = 0
  private_ip                           = "10.0.0.20"
  region                               = "us-east-1"
  source_dest_check                    = true
  subnet_id                            = "subnet-0ac6ed317b471191d"
  tags = {
    Name = "weam-dev-yolo"
  }
  tags_all = {
    Name = "weam-dev-yolo"
  }
  tenancy                     = "default"
  vpc_security_group_ids      = ["sg-09754113d3c1b7524"]
  capacity_reservation_specification {
    capacity_reservation_preference = "open"
  }
  cpu_options {
    core_count       = 1
    threads_per_core = 2
  }
  credit_specification {
    cpu_credits = "unlimited"
  }
  enclave_options {
    enabled = false
  }
  maintenance_options {
    auto_recovery = "default"
  }
  metadata_options {
    http_endpoint               = "enabled"
    http_protocol_ipv6          = "disabled"
    http_put_response_hop_limit = 2
    http_tokens                 = "required"
    instance_metadata_tags      = "disabled"
  }
  private_dns_name_options {
    enable_resource_name_dns_a_record    = false
    enable_resource_name_dns_aaaa_record = false
    hostname_type                        = "ip-name"
  }
  root_block_device {
    delete_on_termination = true
    encrypted             = false
    iops                  = 3000
    throughput            = 125
    volume_size           = 20
    volume_type           = "gp3"
  }
}
*/

removed {
  from = aws_instance.my_ec2

  lifecycle {
    destroy = false
  }
}