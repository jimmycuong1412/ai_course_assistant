# 🎓 AI Course Assistant

A Streamlit-based intelligent chatbot that answers questions about AI course materials using Azure OpenAI. The assistant can reference both preloaded course PDFs and user-uploaded documents to provide accurate, context-aware answers.

## ✨ Features

- **Intelligent Q&A**: Ask questions about course materials and get AI-powered answers
- **Multi-document Support**: Reference course PDFs and upload additional documents
- **Streaming Responses**: Real-time response streaming for better UX
- **Chat History**: Maintain conversation context throughout your session
- **Easy Configuration**: Simple sidebar setup for Azure OpenAI credentials
- **PDF Upload**: Upload additional PDFs on-the-fly for extended context

## 📋 Prerequisites

- **Python 3.7+** - Download from [python.org](https://www.python.org/)
- **pip** - Comes with Python
- **Azure OpenAI Account** - For API access
  - Endpoint URL
  - API Key
  - Model deployment name

## 🚀 Quick Start

### Option 1: Automated (Recommended)

**Windows Command Prompt:**
```bash
start.bat
```

**Windows PowerShell:**
```powershell
.\start.ps1
```

**Linux/macOS or Git Bash:**
```bash
bash start.sh
```

### Option 2: Manual

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## ⚙️ Configuration

### Azure OpenAI Setup

1. Start the application
2. Look for the **Azure OpenAI Settings** panel in the left sidebar
3. Enter your credentials:
   - **Endpoint**: Your Azure OpenAI resource endpoint (e.g., `https://your-resource.openai.azure.com/`)
   - **API Key**: Your Azure OpenAI API key
   - **Model name**: Your deployment name (e.g., `gpt-4`, `gpt-35-turbo`)

### Environment Variables (Optional)

Create a `.env` file in the project root to automatically load credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_MODEL=gpt-4
```

An example `.env.example` file is provided.

## 📚 Course Materials

The app automatically loads PDFs from the `AI Application Engineer Level 1` directory.

### Adding Course Materials

1. Create or place PDF files in the `AI Application Engineer Level 1` folder
2. Restart the app or let it auto-reload
3. The materials will be available for the AI to reference

### Uploading Additional Materials

1. In the app sidebar, find the "Extra PDF Context" section
2. Click "Upload another PDF"
3. Select your PDF file
4. The content will be added to the AI's context for this session

## 💬 How to Use

1. **Start the app** using one of the startup scripts
2. **Configure credentials** in the sidebar
3. **Ask questions** about the course materials in the chat input
4. **View responses** as they stream in real-time
5. **Continue the conversation** - context is maintained throughout
6. **Clear history** - Use the "Clear chat history" button to start fresh

### Example Questions

- "What are the main topics covered in Module 1?"
- "Explain the concept of X as described in the course materials"
- "What are the prerequisites for this course?"
- "Summarize the key points from the lectures"

## 📁 Project Structure

```
workshop1/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── .env                               # Environment variables (gitignored)
├── .env.example                       # Example environment variables
├── .gitignore                         # Git ignore rules
├── start.bat                          # Windows batch startup script
├── start.ps1                          # PowerShell startup script
├── start.sh                           # Bash startup script
├── STARTUP.md                         # Startup guide
├── README.md                          # This file
├── guidelines.txt                     # Project guidelines
└── AI Application Engineer Level 1/   # Course materials directory
    └── *.pdf                          # Course PDF files
```

## 🔧 Troubleshooting

### Python not found
- Ensure Python is installed and added to your PATH
- Restart your terminal after installing Python
- Try using the full path: `C:\Python\python.exe -m streamlit run app.py`

### Dependencies installation fails
- Update pip: `python -m pip install --upgrade pip`
- Try installing with `--user`: `pip install --user -r requirements.txt`
- Check your internet connection

### Azure OpenAI errors
- Verify your endpoint URL is correct (should end with `/`)
- Check that your API key is valid and not expired
- Ensure the model deployment name matches your Azure setup
- Confirm your API key has access to the model

### Port 8501 already in use
- Change the port in the startup command:
  ```bash
  streamlit run app.py --server.port 8502
  ```

### No course materials loaded
- Check the `AI Application Engineer Level 1` directory exists
- Verify PDF files are in the correct location
- Restart the app if you added new PDFs

## 📦 Dependencies

- **streamlit** (>=1.32) - Web app framework
- **openai** (>=1.30) - Azure OpenAI client library
- **python-dotenv** (>=1.0) - Environment variable management
- **pymupdf** (>=1.24) - PDF text extraction

See `requirements.txt` for exact versions.

## 🔒 Security Notes

- **Never commit `.env` file** - It's in `.gitignore` for your safety
- **Keep your API key private** - Don't share it or commit it to version control
- **Use environment variables** - More secure than hardcoding credentials
- **Enable password protection** - If deploying publicly, add authentication

## 🚀 Deployment

For production deployment, consider:

1. **Using environment variables** for all credentials
2. **Adding authentication** (login required)
3. **Setting `server.headless = true`** in `streamlit_config.toml`
4. **Using HTTPS** with a reverse proxy (nginx, etc.)
5. **Rate limiting** for API calls to manage costs
6. **Logging** for monitoring and debugging

## 📝 License

This project is part of the AI Application Engineer training program.

## 👥 Support & Contribution

For issues, questions, or contributions:
1. Check the STARTUP.md for quick troubleshooting
2. Review the guidelines.txt for coding standards
3. Open an issue in the GitLab repository

## 🎯 Next Steps

- [ ] Configure Azure OpenAI credentials
- [ ] Add your course materials to the designated directory
- [ ] Test the chat functionality
- [ ] Explore different question types
- [ ] Customize the system prompt if needed

---

**Happy Learning! 🚀**
