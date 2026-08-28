import type { NextConfig } from 'next';

const config: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@raghub/core', '@raghub/orchestrator'],
};

export default config;