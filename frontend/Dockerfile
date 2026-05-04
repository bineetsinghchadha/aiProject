FROM node:22-alpine AS base
RUN corepack enable && corepack prepare pnpm@latest --activate
WORKDIR /app

FROM base AS deps
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM base AS build
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
ARG API_URL=http://localhost:8080
ENV API_URL=${API_URL}
ARG NEXT_PUBLIC_API_URL=http://localhost:8080
ARG API_URL=http://api:8080
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV API_URL=$API_URL
RUN pnpm build

FROM base AS runner
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=8501
ENV HOSTNAME=0.0.0.0
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 8501
CMD ["node", "server.js"]
# FROM base AS runner
# ENV NODE_ENV=production
# ENV NEXT_TELEMETRY_DISABLED=1
# COPY --from=build /app/.next/standalone ./
# COPY --from=build /app/.next/static ./.next/static
# COPY --from=build /app/public ./public
# EXPOSE 3000
# # ENV PORT=3000  # removed: set by docker-compose
# CMD ["node", "server.js"]
