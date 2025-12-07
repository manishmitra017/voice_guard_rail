import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import { Construct } from 'constructs';

export interface VoiceEmotionStackProps extends cdk.StackProps {
  /**
   * Use spot instances for cost savings (~70% cheaper)
   * Risk: Can be interrupted with 2-min notice
   * @default true
   */
  useSpotInstances?: boolean;

  /**
   * Instance type - t3.medium (4GB) or t3.large (8GB)
   * t3.medium is tighter but cheaper
   * @default t3.medium
   */
  instanceType?: string;

  /**
   * Your domain name for the app (optional)
   * If not provided, uses EC2 public IP
   */
  domainName?: string;
}

export class VoiceEmotionStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: VoiceEmotionStackProps) {
    super(scope, id, props);

    const useSpot = props?.useSpotInstances ?? true;
    const instanceTypeStr = props?.instanceType ?? 't3.medium';

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

    // Allow HTTP/HTTPS and Streamlit port
    securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      'SSH access'
    );
    securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'HTTP access'
    );
    securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(443),
      'HTTPS access'
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

      // Update system
      'yum update -y',
      'yum install -y git docker python3.11 python3.11-pip',

      // Install uv
      'curl -LsSf https://astral.sh/uv/install.sh | sh',
      'export PATH="$HOME/.local/bin:$PATH"',
      'echo \'export PATH="$HOME/.local/bin:$PATH"\' >> /etc/profile.d/uv.sh',

      // Add swap for memory-constrained instances (4GB swap)
      'dd if=/dev/zero of=/swapfile bs=1M count=4096',
      'chmod 600 /swapfile',
      'mkswap /swapfile',
      'swapon /swapfile',
      'echo "/swapfile swap swap defaults 0 0" >> /etc/fstab',

      // Clone the repository
      'cd /home/ec2-user',
      'git clone https://github.com/manishmitra017/voice_guard_rail.git',
      'cd voice_guard_rail',
      'chown -R ec2-user:ec2-user /home/ec2-user/voice_guard_rail',

      // Install dependencies with uv
      'sudo -u ec2-user /root/.local/bin/uv sync',

      // Create systemd service
      'cat > /etc/systemd/system/voice-emotion.service << \'EOF\'',
      '[Unit]',
      'Description=Voice Emotion Detector Streamlit App',
      'After=network.target',
      '',
      '[Service]',
      'Type=simple',
      'User=ec2-user',
      'WorkingDirectory=/home/ec2-user/voice_guard_rail',
      'Environment="PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin"',
      'ExecStart=/home/ec2-user/.local/bin/uv run streamlit run app_cloud.py --server.port 8501 --server.address 0.0.0.0 --server.headless true',
      'Restart=always',
      'RestartSec=10',
      '',
      '[Install]',
      'WantedBy=multi-user.target',
      'EOF',

      // Enable and start service
      'systemctl daemon-reload',
      'systemctl enable voice-emotion',
      'systemctl start voice-emotion',

      // Install and configure nginx as reverse proxy (optional, for port 80)
      'amazon-linux-extras install nginx1 -y || yum install nginx -y',
      'cat > /etc/nginx/conf.d/voice-emotion.conf << \'EOF\'',
      'server {',
      '    listen 80;',
      '    server_name _;',
      '',
      '    location / {',
      '        proxy_pass http://localhost:8501;',
      '        proxy_http_version 1.1;',
      '        proxy_set_header Upgrade $http_upgrade;',
      '        proxy_set_header Connection "upgrade";',
      '        proxy_set_header Host $host;',
      '        proxy_set_header X-Real-IP $remote_addr;',
      '        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;',
      '        proxy_set_header X-Forwarded-Proto $scheme;',
      '        proxy_read_timeout 86400;',
      '    }',
      '}',
      'EOF',
      'systemctl enable nginx',
      'systemctl start nginx',
    );

    // Launch Template
    const launchTemplate = new ec2.LaunchTemplate(this, 'VoiceEmotionLT', {
      instanceType: new ec2.InstanceType(instanceTypeStr),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      securityGroup,
      role,
      userData,
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(20, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
          }),
        },
      ],
      // Spot configuration - must use ONE_TIME for Auto Scaling Groups
      ...(useSpot && {
        spotOptions: {
          requestType: ec2.SpotRequestType.ONE_TIME,
          maxPrice: 0.05, // Max $0.05/hour (t3.medium spot is ~$0.01)
        },
      }),
    });

    // Auto Scaling Group (single instance)
    const asg = new autoscaling.AutoScalingGroup(this, 'VoiceEmotionASG', {
      vpc,
      launchTemplate,
      minCapacity: 1,
      maxCapacity: 1,
      desiredCapacity: 1,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PUBLIC,
      },
      healthCheck: autoscaling.HealthCheck.ec2({
        grace: cdk.Duration.minutes(10), // Allow time for model download
      }),
    });

    // Elastic IP for stable address (used by CloudFront origin)
    const eip = new ec2.CfnEIP(this, 'VoiceEmotionEIP', {
      domain: 'vpc',
      tags: [{ key: 'Name', value: 'VoiceEmotionDetector' }],
    });

    // CloudFront Distribution for HTTPS access
    const distribution = new cloudfront.Distribution(this, 'VoiceEmotionCDN', {
      defaultBehavior: {
        origin: new origins.HttpOrigin(eip.attrPublicIp, {
          protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
          httpPort: 80,
        }),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
        originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
      },
      // WebSocket support for Streamlit
      additionalBehaviors: {
        '/_stcore/*': {
          origin: new origins.HttpOrigin(eip.attrPublicIp, {
            protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
            httpPort: 80,
          }),
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
        },
      },
      priceClass: cloudfront.PriceClass.PRICE_CLASS_100, // Cheapest - US, Canada, Europe
      comment: 'Voice Emotion Detector CDN',
    });

    // Only output the CloudFront URL (public-facing)
    new cdk.CfnOutput(this, 'AppURL', {
      value: `https://${distribution.distributionDomainName}`,
      description: 'Voice Emotion Detector URL',
    });

    // Tag all resources
    cdk.Tags.of(this).add('Project', 'VoiceEmotionDetector');
    cdk.Tags.of(this).add('Environment', 'Production');
    cdk.Tags.of(this).add('CostCenter', 'Personal');
  }
}
