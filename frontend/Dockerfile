###############################################################################
# Vigilia — Frontend Dockerfile
# Multi-stage: desarrollo (dev) y producción (prod con Nginx)
###############################################################################

# =============================================================================
# Stage 1: Base con Node.js
# =============================================================================
FROM node:20-alpine AS base

WORKDIR /app

# Copiar archivos de dependencias
COPY package.json package-lock.json* ./

# Instalar dependencias
RUN npm install

# Copiar todo el código fuente
COPY . .

# Puerto del dev server
EXPOSE 5173

# Comando de desarrollo (Vite con HMR)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# =============================================================================
# Stage 2: Build para producción
# =============================================================================
FROM base AS build

RUN npm run build

# =============================================================================
# Stage 3: Producción con Nginx
# =============================================================================
FROM nginx:alpine AS production

# Copiar configuración personalizada de Nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar los archivos compilados desde el stage de build
COPY --from=build /app/dist /usr/share/nginx/html

# Puerto de producción
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
