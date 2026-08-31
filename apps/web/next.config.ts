import type { NextConfig } from 'next';

const config: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@revex/core', '@revex/orchestrator'],
};

export default config;