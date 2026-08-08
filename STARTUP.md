# Getting Started - AI Course Assistant

This project is a Streamlit-based chatbot that answers questions about AI course materials using Azure OpenAI.

## Quick Start

Choose one of the startup scripts below based on your shell/terminal:

### Windows Command Prompt
```cmd
start.bat
```

### Windows PowerShell
```powershell
.\start.ps1
```

Or with no dependency installation:
```powershell
.\start.ps1 -NoInstall
```

### Git Bash or Linux/macOS
```bash
bash start.sh
```

## What the Scripts Do

1. **Check Python installation** - Verifies Python is available
2. **Install dependencies** - Runs `pip install -r requirements.txt`
3. **Start Streamlit** - Launches the web app on `http://localhost:8501`

## Prerequisites

- **Python 3.7+** installed and in your PATH
- **pip** package manager

## Configuration

Before using the app, you'll need to configure Azure OpenAI credentials:

1. Open the app at `http://localhost:8501`
2. In the left sidebar, fill in:
   - **Endpoint**: Your Azure OpenAI endpoint URL
   - **API Key**: Your Azure OpenAI API key
   - **Model name**: The deployment name (e.g., `gpt-4`)

You can also set these as environment variables in a `.env` file:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_MODEL=gpt-4
```

## Course Materials

The app loads PDFs from the `AI Application Engineer Level 1` directory. Place course materials there as PDF files.

You can also upload additional PDFs directly in the app sidebar under "Extra PDF Context".

## Troubleshooting

- **Python not found**: Make sure Python is installed and added to your PATH
- **Dependencies fail**: Try `python -m pip install --upgrade pip` then run the start script again
- **Azure credentials error**: Verify your endpoint, API key, and model name in the sidebar
- **Port 8501 in use**: Change the port by modifying `streamlit run app.py --server.port 8502`
