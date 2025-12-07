#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { VoiceEmotionStack } from '../lib/voice-emotion-stack';

const app = new cdk.App();

// Voice Emotion Detector Stack
// Uses EC2 Spot instances for cost optimization (~$12-15/month)
new VoiceEmotionStack(app, 'VoiceEmotionStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'ap-southeast-2',
  },

  // Cost optimization settings
  useSpotInstances: true,      // 70% cheaper than on-demand
  instanceType: 't3.medium',   // 4GB RAM + 4GB swap = enough for models

  // Stack description
  description: 'Voice Emotion Detector - Real-time speech emotion recognition using AI',
});

app.synth();
