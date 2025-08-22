import requests
import sys
from rich.console import Console
from llama_cpp import Llama
from contextlib import redirect_stderr
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn

console = Console()


url = "https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_0.gguf"
output_path = "LFM2-1.2B-Q4_0.gguf"

response = requests.get(url, stream=True)
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("✅ Download complete:", output_path)

SYSTEM_PROMPT = "You're a helpful assistant. Be concise and accurate."
MODEL_GGUF_FILE = "./LFM2-1.2B-Q4_0.gguf"

# Initialize FastAPI
app = FastAPI()

# Load model once at startup
console.print(
    f"[yellow]⏳ Loading model:[/yellow] [dim]{MODEL_GGUF_FILE}[/dim]")
llm = Llama(model_path=MODEL_GGUF_FILE)
console.print("[green]✅ Model loaded successfully[/green]")

# Shared chat history (global for now)
history = [{"role": "system", "content": SYSTEM_PROMPT}]


@app.get("/generate")
def generate(prompt: str = Query(..., description="User's input prompt")):
    if not prompt.strip():
        return JSONResponse(content={"error": "Prompt cannot be empty"}, status_code=400)

    # Append user message
    history.append({"role": "user", "content": prompt})

    # Generate assistant response
    output = llm.create_chat_completion(messages=history)
    generated_message = output["choices"][0]["message"]["content"]

    # Save assistant response to history
    history.append({"role": "assistant", "content": generated_message})

    return JSONResponse(content={"response": generated_message})


if __name__ == "__main__":
    with open("llamacpp.log", "w") as f:
        with redirect_stderr(f):
            try:
                uvicorn.run(app, host="127.0.0.1", port=8000)
            except Exception as e:
                console.print(f"\n[red]❌ Error: {e}[/red]")
                sys.exit(1)
