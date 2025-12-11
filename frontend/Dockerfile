# ============================================================================
# Multi-stage build: Frontend React/Vite application
# ============================================================================

# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json bun.lockb* ./

# Install dependencies with npm (compatibility, bun optional)
RUN npm install

# Copy source code
COPY . .

# Build production bundle
RUN npm run build

# Stage 2: Production (serve with lightweight server)
FROM node:20-alpine

WORKDIR /app

# Install serve package to host static files
RUN npm install -g serve

# Copy built app from builder stage
COPY --from=builder /app/dist ./dist

# Expose port for Vite dev server or serve
EXPOSE 5173

# Default: Start production server
CMD ["serve", "-s", "dist", "-l", "5173"]

# Alternative: For development with hot reload, use:
# CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
