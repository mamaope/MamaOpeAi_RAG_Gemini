provider "aws" {
  region = "us-east-1"
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# Create a VPC
resource "aws_vpc" "mamaopeai_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "MamaOpeAI VPC"
  }
}

# Create an Internet Gateway
resource "aws_internet_gateway" "mamaopeai_igw" {
  vpc_id = aws_vpc.mamaopeai_vpc.id

  tags = {
    Name = "MamaOpeAI Internet Gateway"
  }
}

# Create a public subnet
resource "aws_subnet" "mamaopeai_subnet" {
  vpc_id                  = aws_vpc.mamaopeai_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "us-east-1a"

  tags = {
    Name = "MamaOpeAI Public Subnet"
  }
}

# Create a Route Table
resource "aws_route_table" "mamaopeai_route_table" {
  vpc_id = aws_vpc.mamaopeai_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mamaopeai_igw.id
  }

  tags = {
    Name = "MamaOpeAI Route Table"
  }
}

# Associate the Route Table with the Subnet
resource "aws_route_table_association" "mamaopeai_route_table_assoc" {
  subnet_id      = aws_subnet.mamaopeai_subnet.id
  route_table_id = aws_route_table.mamaopeai_route_table.id
}

# Security Group
resource "aws_security_group" "mamaopeai_sg" {
    vpc_id = aws_vpc.mamaopeai_vpc.id

    ingress {
    description = "Allow HTTP traffic"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    }

    ingress {
    description = "Allow HTTPS traffic"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    }

    ingress {
    description = "Allow FastAPI port"
    from_port   = 8090
    to_port     = 8090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    }

    ingress {
    description = "Allow SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    }

    egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    }

    tags = {
    Name = "MamaOpeAI Security Group"
    }
}

# Create an EC2 Instance
resource "aws_instance" "mamaopeai" {
    ami           = "ami-0e2c8caa4b6378d8c"
    instance_type = "t2.micro"
    subnet_id     = aws_subnet.mamaopeai_subnet.id

    # IAM Role for S3 access
    iam_instance_profile = aws_iam_instance_profile.ec2_instance_profile.name

    # Security Group Association
    vpc_security_group_ids = [aws_security_group.mamaopeai_sg.id]

    # Key Pair for SSH Access
    key_name = "mamaopeai"

    # User Data for Initialization
    user_data = <<-EOF
      #!/bin/bash
      apt-get update -y
      apt-get install -y docker.io

      # install aws cli
      sudo apt install unzip
      curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
      unzip awscliv2.zip
      sudo ./aws/install

      usermod -aG docker ubuntu
      systemctl enable docker
      systemctl start docker

      # Fetch the .env file from S3
      aws s3 cp s3://${aws_s3_bucket.env_bucket.bucket}/.env /home/ubuntu/.env

      # Pull the latest Docker image from Docker Hub
      docker pull richkitibwa/mamaopeai-api:latest

      # Stop and remove any existing containers
      docker stop mamaopeai-api || true
      docker rm mamaopeai-api || true

      # Run the Dockerized FastAPI app with environment variables
      docker run -d --name mamaopeai-api --restart always --env-file /home/ubuntu/.env -p 8090:8090 richkitibwa/mamaopeai-api:latest
    EOF

    tags = {
    Name = "MamaOpeAI-API"
  }            
}

# Elastic IP for Static IP Address
resource "aws_eip" "mamaopeai_eip" {
  domain = "vpc"
  tags = {
    Name = "MamaOpeAI Elastic IP"
  }
  lifecycle {
    prevent_destroy = true
  }
}

# Associate Elastic IP with the EC2 Instance
resource "aws_eip_association" "mamaopeai_eip_assoc" {
  instance_id   = aws_instance.mamaopeai.id
  allocation_id = aws_eip.mamaopeai_eip.id
}

# S3 Bucket for .env
resource "aws_s3_bucket" "env_bucket" {
  bucket = "mamaopeai-api-env-bucket"

  tags = {
    Name        = "MamaOpe Env Bucket"
  }
}  

# Upload the .env file to the S3 bucket
resource "aws_s3_object" "env_file" {
  bucket = aws_s3_bucket.env_bucket.id
  key    = ".env"
  source = "../.env"
  acl    = "private"
}

# IAM Role for EC2 to access the S3 bucket
resource "aws_iam_role" "ec2_role" {
  name = "ec2_s3_access_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}


# Attach a policy to the IAM Role for S3 access
resource "aws_iam_policy" "ec2_s3_access_policy" {
  name   = "ec2_s3_access_policy"
  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:HeadBucket",
          "s3:PutObject",
          "s3:GetBucketLocation"
        ],
        Resource = [
          "arn:aws:s3:::${aws_s3_bucket.env_bucket.bucket}",
          "arn:aws:s3:::${aws_s3_bucket.env_bucket.bucket}/*"
        ]
      }
    ]
  })
}

# Attach the IAM policy to the EC2 role
resource "aws_iam_role_policy_attachment" "ec2_s3_access_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_s3_access_policy.arn
}

# Attach the IAM Role to an Instance Profile
resource "aws_iam_instance_profile" "ec2_instance_profile" {
  name = "ec2_s3_instance_profile"
  role = aws_iam_role.ec2_role.name
}
