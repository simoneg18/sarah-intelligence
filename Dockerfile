FROM python:3.12-slim

# Install system dependencies: ffmpeg + yt-dlp
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp && \
    chmod a+rx /usr/local/bin/yt-dlp && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY youtube_agent.py .
COPY generate_sarah_guide.py .

# Create output directory
RUN mkdir -p /app/output

ENV OUTPUT_DIR=/app/output
ENV PORT=8787

EXPOSE 8787

CMD ["python", "-u", "youtube_agent.py", "serve"]
