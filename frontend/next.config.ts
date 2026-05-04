import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    proxyTimeout: 600_000,
  },
  async rewrites() {
    // API_URL is evaluated at RUNTIME by Next.js standalone server,
    // not baked at build time — so http://api:8080 works inside Docker.
    const apiUrl = process.env.API_URL ?? "http://localhost:8080";
    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;