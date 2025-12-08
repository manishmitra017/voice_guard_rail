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

    // Allow HTTP and Streamlit port
    securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'HTTP access'
    );
    securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(8501),
      'Streamlit access'
    );

    // IAM Role for EC2
    const role = new iam.Role(this, 'VoiceEmotionRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // User data script to set up the application
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'set -e',
      'exec > >(tee /var/log/user-data.log) 2>&1',

      // Update system
      'yum update -y',
      'yum install -y git python3.11 python3.11-pip nginx',

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

      // Install dependencies with uv as ec2-user
      'cd /home/ec2-user/voice_guard_rail',
      'sudo -u ec2-user /home/ec2-user/.local/bin/uv sync',

      // Create systemd service
      `cat > /etc/systemd/system/voice-emotion.service << 'SERVICEEOF'
[Unit]
Description=Voice Emotion Detector Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/voice_guard_rail
Environment="PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ec2-user/.local/bin/uv run streamlit run app_cloud.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF`,

      // Enable and start service
      'systemctl daemon-reload',
      'systemctl enable voice-emotion',
      'systemctl start voice-emotion',

      // Configure nginx as reverse proxy for port 80
      `cat > /etc/nginx/conf.d/voice-emotion.conf << 'NGINXEOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
NGINXEOF`,
      'rm -f /etc/nginx/conf.d/default.conf',
      'systemctl enable nginx',
      'systemctl start nginx',
    );

    // Use L2 Instance construct - more reliable than CfnInstance with LaunchTemplate
    // For Spot instances, we use LaunchTemplate approach with proper lifecycle
    const instance = new ec2.Instance(this, 'VoiceEmotionInstance', {
      vpc,
      instanceType: new ec2.InstanceType(instanceTypeStr),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      securityGroup,
      role,
      userData,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(20, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
          }),
        },
      ],
    });

    // Apply Spot configuration via escape hatch if requested
    // Using one-time Spot request - simpler and more reliable
    if (useSpot) {
      const cfnInstance = instance.node.defaultChild as ec2.CfnInstance;
      cfnInstance.addPropertyOverride('InstanceMarketOptions', {
        MarketType: 'spot',
        SpotOptions: {
          MaxPrice: '0.05', // Max $0.05/hour (t3.medium spot is ~$0.02)
        },
      });
    }

    // Elastic IP for stable address
    const eip = new ec2.CfnEIP(this, 'VoiceEmotionEIP', {
      domain: 'vpc',
      instanceId: instance.instanceId,
      tags: [{ key: 'Name', value: 'VoiceEmotionDetector' }],
    });

    // Ensure EIP is created after instance
    eip.addDependency(instance.node.defaultChild as cdk.CfnResource);

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
