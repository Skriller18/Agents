# Multimodal Live API Console

A comprehensive application that enables integration with Google's Multimodal Live API and Gemini models, providing real-time audio/video streaming, interactive AI conversations, visualization capabilities, and AI agent features.

![Multimodal Live API Console]

## 🌟 Features

- **Real-time streaming communication** with Gemini models
- **Audio and video input/output** for multimodal conversations
- **Screen sharing and webcam capture** for visual context
- **Dynamic visualization** using Altair/Vega charts
- **Tool-based interaction** with Google Search integration
- **Intelligent math tutoring** features
- **Interactive console UI** for tracking conversations and tools usage
- **AI agents** that can interact with your computer screen

## 📦 Components

This repository includes:

1. **Web Console Application** - React-based frontend interface
2. **Gemini Live API Integration** - Python scripts for audio/video streaming with Gemini
3. **Claude Computer Use** - Tools for Claude AI to interact with your computer
4. **LiveKit Agent Integration** - Video conferencing with AI agents

## 📋 Prerequisites

- Node.js v16+ and npm
- Python 3.9+
- Google Cloud API key with Gemini API access
- LiveKit credentials (for LiveKit agent features)
- Anthropic API key (for Claude computer use features)

## 🚀 Getting Started

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd multimodal-live-api-web-console
   ```

2. **Set up the web application**:
   ```bash
   npm install
   ```

3. **Install Python dependencies** (for Gemini and LiveKit scripts):
   ```bash
   pip install -r requirements.txt
   pip install -r claude_computer_use/requirements.txt
   ```

4. **Configure environment variables**:
   
   Create a `.env` file in the root directory:
   ```
   REACT_APP_GEMINI_API_KEY=your_gemini_api_key
   NEXT_PUBLIC_LIVEKIT_URL=your_livekit_url
   LIVEKIT_API_KEY=your_livekit_api_key
   LIVEKIT_API_SECRET=your_livekit_api_secret
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

### Running the Application

1. **Start the web console**:
   ```bash
   npm start
   ```
   The application will be available at http://localhost:3000

2. **Run Gemini live API scripts** (in a separate terminal):
   ```bash
   python gemini-live.py
   ```

3. **Run Claude computer use** (in a separate terminal, if needed):
   ```bash
   python claude_computer_use/main.py "Your instruction here"
   ```

4. **Run LiveKit video agent** (in a separate terminal, if needed):
   ```bash
   python livekit-openai-agent.py
   ```

## 🔧 Usage

### Web Console

1. Click the "Play" button to connect to the Gemini API
2. Use the microphone button to enable/disable audio input
3. Use the screen sharing or webcam buttons to enable visual context
4. Type messages in the input box or use your microphone to converse with the AI
5. View conversation history and tool usage in the sidebar

### Gemini Live Python Script

```bash
# For camera mode (default)
python gemini-live.py

# For screen sharing mode
python gemini-live.py --mode screen

# Without video
python gemini-live.py --mode none
```

### Claude Computer Use

```bash
python claude_computer_use/main.py "Save an image of a cat to the desktop"
```

## 📊 Visualization Features

The application includes integration with Altair/Vega for creating visualizations:

1. Ask the AI to create a chart or graph
2. The AI will use the `render_altair` function to generate the visualization
3. Results will be displayed in the main application area

## 👩‍🏫 Math Tutoring Features

Use the math tutoring capabilities:

1. Share your screen showing math problems or equations
2. The AI will use the `check_work` function to analyze your work
3. Get step-by-step feedback on your solutions

## 🔍 Tool Integration

The application integrates with Google Search and other tools:

1. Ask questions that require up-to-date information
2. View search results within the conversation
3. Use specialized functions for specific tasks like checking mathematical work

## 🚢 Deployment

To deploy the application to Google App Engine:

```bash
gcloud app deploy app.yaml
```

## 🧰 Project Structure

- `/src` - React application source code
- `/public` - Static assets
- `/src/components` - React components
- `/src/contexts` - React contexts including LiveAPIContext
- `/src/hooks` - Custom React hooks
- `/src/lib` - Utility functions and classes
- `/claude_computer_use` - Claude AI computer interaction tools
- Python scripts in the root directory for various AI agent capabilities

## 🔒 Security Notes

- API keys should never be committed to version control
- Use environment variables for sensitive configuration
- Ensure proper authentication for any deployed instances

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Additional Resources

- [Google Generative AI Documentation](https://ai.google.dev/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [LiveKit Documentation](https://docs.livekit.io/)
