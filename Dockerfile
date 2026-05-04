# FROM python:3.11-slim
# WORKDIR /workspace
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# FROM python:3.11-slim

# # Install system deps in one layer, clean up in same RUN to keep layer small
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /workspace

# # Copy requirements first — this layer is cached unless requirements.txt changes
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Non-root user for security
# RUN useradd -m appuser && chown -R appuser:appuser /workspace
# USER appuser

# # Source code is mounted as a volume at runtime, not copied in
# # so no COPY . . needed here

# Stage 1 — builder: installs everything including build tools
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2 — runtime: copies only the installed packages, not the build tools
FROM python:3.11-slim

WORKDIR /workspace
COPY --from=builder /install /usr/local

RUN useradd -m appuser && chown -R appuser:appuser /workspace
USER appuser