#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { VoiceEmotionStack } from '../lib/voice-emotion-stack';

const app = new cdk.App();

// Voice Emotion Detector Stack
new VoiceEmotionStack(app, 'VoiceEmotionStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'ap-southeast-2',
  },
  instanceType: 't3.medium',     // 4GB RAM + 4GB swap = enough for models
  useSpotInstances: false,       // Set to true for ~70% savings, but requires instance recreation
  description: 'Voice Emotion Detector - Real-time speech emotion recognition using AI',
});

app.synth();
