import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface VoiceEmotionStackProps extends cdk.StackProps {
  /**
   * Instance type - t3.medium (4GB) recommended
   * @default t3.medium
   */
  instanceType?: string;

  /**
   * Use spot instances for cost savings (~70% cheaper)
   * @default true
   */
  useSpotInstances?: boolean;
}

export class VoiceEmotionStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: VoiceEmotionStackProps) {
    super(scope, id, props);

    const instanceTypeStr = props?.instanceType ?? 't3.medium';
    const useSpot = props?.useSpotInstances ?? true;

    // VPC - Use default VPC to save costs (no NAT Gateway)
    const vpc = ec2.Vpc.fromLookup(this, 'DefaultVPC', {
      isDefault: true,
    });

    // Security Group
    const securityGroup = new ec2.SecurityGroup(this, 'VoiceEmotionSG', {
      vpc,
      description: 'Security group for Voice Emotion Detector',
      allowAllOutbound: true,
    });

    // Allow HTTP access
    securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'HTTP access'
    );

    // IAM Role for EC2
    const role = new iam.Role(this, 'VoiceEmotionRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // Instance Profile for the role
    const instanceProfile = new iam.CfnInstanceProfile(this, 'VoiceEmotionProfile', {
      roles: [role.roleName],
    });

    // User data script - React + FastAPI setup
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'set -e',
      'exec > >(tee /var/log/user-data.log) 2>&1',
      'echo "Starting setup at $(date)"',

      // Update system
      'yum update -y',
      'yum install -y git python3.11 python3.11-pip nginx',

      // Install Node.js 20 for React build
      'curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -',
      'yum install -y nodejs',

      // Install uv for ec2-user
      'sudo -u ec2-user bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"',

      // Add swap for memory-constrained instances (4GB swap)
      'dd if=/dev/zero of=/swapfile bs=1M count=4096',
      'chmod 600 /swapfile',
      'mkswap /swapfile',
      'swapon /swapfile',
      'echo "/swapfile swap swap defaults 0 0" >> /etc/fstab',

      // Clone the repository
      'cd /home/ec2-user',
      'git clone https://github.com/manishmitra017/voice_guard_rail.git',
      'chown -R ec2-user:ec2-user /home/ec2-user/voice_guard_rail',

      // Install Python dependencies with uv
      'cd /home/ec2-user/voice_guard_rail',
      'sudo -u ec2-user /home/ec2-user/.local/bin/uv sync',

      // Build React frontend
      'cd /home/ec2-user/voice_guard_rail/frontend',
      'npm install',
      'npm run build',

      // Copy built frontend to nginx directory
      'mkdir -p /var/www/voice-emotion',
      'cp -r /home/ec2-user/voice_guard_rail/frontend/dist/* /var/www/voice-emotion/',

      // Create FastAPI systemd service
      `cat > /etc/systemd/system/voice-emotion-api.service << 'SERVICEEOF'
[Unit]
Description=Voice Emotion Detector FastAPI
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/voice_guard_rail
Environment="PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ec2-user/.local/bin/uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF`,

      // Configure nginx - serve React static + proxy /api to FastAPI
      `cat > /etc/nginx/conf.d/voice-emotion.conf << 'NGINXEOF'
server {
    listen 80;
    server_name _;

    # Serve React frontend
    root /var/www/voice-emotion;
    index index.html;

    # React SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to FastAPI
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;

        # Handle large audio uploads
        client_max_body_size 50M;
    }

    # Static file caching
    location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
NGINXEOF`,

      // Remove default nginx config
      'rm -f /etc/nginx/conf.d/default.conf',
      'rm -f /etc/nginx/sites-enabled/default',

      // Enable and start services
      'systemctl daemon-reload',
      'systemctl enable voice-emotion-api',
      'systemctl start voice-emotion-api',
      'systemctl enable nginx',
      'systemctl start nginx',

      'echo "Setup complete at $(date)"',
    );

    // Get latest Amazon Linux 2023 AMI
    const ami = ec2.MachineImage.latestAmazonLinux2023();
    const amiId = ami.getImage(this).imageId;

    // Create Launch Template with Spot configuration
    const launchTemplate = new ec2.CfnLaunchTemplate(this, 'VoiceEmotionLT', {
      launchTemplateData: {
        instanceType: instanceTypeStr,
        imageId: amiId,
        iamInstanceProfile: {
          arn: instanceProfile.attrArn,
        },
        securityGroupIds: [securityGroup.securityGroupId],
        blockDeviceMappings: [
          {
            deviceName: '/dev/xvda',
            ebs: {
              volumeSize: 20,
              volumeType: 'gp3',
              encrypted: true,
              deleteOnTermination: true,
            },
          },
        ],
        userData: cdk.Fn.base64(userData.render()),
        // Spot configuration
        ...(useSpot && {
          instanceMarketOptions: {
            marketType: 'spot',
            spotOptions: {
              maxPrice: '0.05',
              spotInstanceType: 'one-time',
            },
          },
        }),
        tagSpecifications: [
          {
            resourceType: 'instance',
            tags: [
              { key: 'Name', value: 'VoiceEmotionDetector' },
              { key: 'Project', value: 'VoiceEmotionDetector' },
            ],
          },
        ],
      },
    });

    // Ensure launch template is created after instance profile
    launchTemplate.addDependency(instanceProfile);

    // Try different AZs for Spot capacity - prefer cheapest AZ
    const preferredSubnet = vpc.publicSubnets.find(s =>
      s.availabilityZone.endsWith('a')
    ) || vpc.publicSubnets[0];

    // Create EC2 instance using Launch Template
    const instance = new ec2.CfnInstance(this, 'VoiceEmotionInstance', {
      launchTemplate: {
        launchTemplateId: launchTemplate.ref,
        version: launchTemplate.attrLatestVersionNumber,
      },
      subnetId: preferredSubnet.subnetId,
    });

    // Ensure instance is created after launch template
    instance.addDependency(launchTemplate);

    // Elastic IP for stable address
    const eip = new ec2.CfnEIP(this, 'VoiceEmotionEIP', {
      domain: 'vpc',
      instanceId: instance.ref,
      tags: [{ key: 'Name', value: 'VoiceEmotionDetector' }],
    });

    // Ensure EIP is created after instance
    eip.addDependency(instance);

    // Output only the app URL
    new cdk.CfnOutput(this, 'AppURL', {
      value: `http://${eip.attrPublicIp}`,
      description: 'Voice Emotion Detector URL',
    });

    // Tag all resources
    cdk.Tags.of(this).add('Project', 'VoiceEmotionDetector');
    cdk.Tags.of(this).add('Environment', 'Production');
  }
}
