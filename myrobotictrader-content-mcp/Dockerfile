FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY myrobotictrader_content_mcp.py .

ENV MCP_TRANSPORT=sse
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8084
ENV DATA_DIR=/data

EXPOSE 8084

CMD ["python", "myrobotictrader_content_mcp.py"]
