# AI Agents Playground

A collaborative repository focused on building and experimenting with AI agents. This project integrates with Google's Multimodal Live API and Gemini models, Anthropic's Claude, and LiveKit to create interactive AI agents with real-time audio/video streaming capabilities, computer control features, and multimodal interactions.

![Multimodal Live API Console](https://via.placeholder.com/800x400)

## 🌟 Features

- **Real-time streaming communication** with Gemini and Claude models
- **Audio and video input/output** for multimodal conversations
- **Screen sharing and webcam capture** for visual context
- **Computer control capabilities** allowing agents to interact with your desktop
- **Dynamic visualization** using Altair/Vega charts
- **Tool-based interaction** with Google Search integration
- **Intelligent math tutoring** features
- **Interactive console UI** for tracking conversations and tools usage

Feel free to extend these capabilities or create entirely new agent types!

## 🔍 Agent Capabilities

This repository contains implementations of several AI agent types:

1. **Multimodal Web Agents** - React-based agents that can process and respond with text, audio, and visuals
2. **Computer-Controlling Agents** - Claude-powered agents that can interact with your desktop
3. **Educational Math Tutoring Agents** - Agents that can analyze and provide feedback on written math work
4. **Video Conference Agents** - LiveKit-integrated agents for real-time video interaction

Each agent type demonstrates different capabilities and integration approaches, providing a valuable resource for anyone interested in building their own AI agents.

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

## 🔧 Usage Examples

### Web Console Agent

```bash
# Start the web application
npm start

# In your browser at http://localhost:3000
# 1. Click the "Play" button to connect to the Gemini API
# 2. Use the microphone button to enable/disable audio input
# 3. Use the screen sharing or webcam buttons to enable visual context
# 4. Type messages or use your microphone to converse with the AI
# 5. View conversation history and tool usage in the sidebar
```

### Gemini Live Agent

```bash
# For camera mode (default)
python gemini-live.py

# For screen sharing mode
python gemini-live.py --mode screen

# Without video
python gemini-live.py --mode none
```

### Claude Computer-Controlling Agent

```bash
# Give Claude instructions to perform on your computer
python claude_computer_use/main.py "Search for the latest AI news and take a screenshot"

# Try more complex instructions
python claude_computer_use/main.py "Create a new folder on the desktop named 'AI Projects'"
```

### LiveKit Video Agent

```bash
# Start the math tutor video agent
python livekit-openai-agent.py

# Connect to a LiveKit room to interact with the agent via video
```

Feel free to experiment with different instructions and capabilities!

## 🔬 Future Agent Explorations

Here are some exciting directions to explore with this codebase:

- **Cross-model agent collaboration**: Create systems where Gemini and Claude agents collaborate
- **Specialized domain agents**: Build agents optimized for specific domains (healthcare, education, etc.)
- **Agent memory systems**: Implement better memory and context management for longer interactions
- **Multimodal reasoning**: Enhance agents' ability to reason across text, images, audio, and actions
- **Agent autonomy levels**: Experiment with different autonomy/supervision balances
- **Multi-agent systems**: Create environments where multiple agents interact with each other
- **Agent personalization**: Build systems that adapt to individual user preferences and needs

If you implement any of these ideas, please contribute back to the repository!

## 🧠 Agent Architecture

Each agent in this repository follows a similar architecture pattern:

1. **Input Processing** - Capturing and processing user inputs (text, audio, video)
2. **AI Model Integration** - Connecting to AI models (Gemini, Claude) via their respective APIs
3. **Tool Integration** - Enabling the AI to use tools (search, visualization, computer control)
4. **Output Generation** - Producing multimodal responses (text, speech, visual elements)
5. **Feedback Loop** - Maintaining context and enabling iterative interactions

These components can be mixed and matched to create new types of agents with different capabilities.

Feel free to experiment with the architecture and create your own agent variants!

## 🚢 Deployment Options

The repository components can be deployed in various ways:

### Web Console
Deploy to Google App Engine:
```bash
gcloud app deploy app.yaml
```

### Agent Services
For production deployments, consider:
- Containerizing agents with Docker
- Deploying to cloud services (GCP, AWS, Azure)
- Setting up CI/CD pipelines for automated deployment

Feel free to contribute deployment templates and examples!

## 🔒 Security Notes

- API keys should never be committed to version control
- Use environment variables for sensitive configuration
- Ensure proper authentication for any deployed instances
- Be mindful of permissions when using computer-controlling agents
- Review any code generated or modified by agents before execution

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🤝 Contributing

This repository is open for community contributions! Feel free to:

- **Explore the code**: Understand how different AI agent systems work
- **Report issues**: Open an issue if you find bugs or have feature suggestions
- **Submit pull requests**: Fix bugs or add new agent capabilities
- **Share your experiments**: Create examples of agent use cases
- **Improve documentation**: Help make the codebase more accessible to others

All contributions, big or small, are welcome. Together we can build better AI agent systems!

### How to Submit a Pull Request

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-agent`)
3. Commit your changes (`git commit -m 'Add some amazing agent feature'`)
4. Push to the branch (`git push origin feature/amazing-agent`)
5. Open a Pull Request

## 📚 Additional Resources

- [Google Generative AI Documentation](https://ai.google.dev/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [LiveKit Documentation](https://docs.livekit.io/)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [WebRTC Documentation](https://webrtc.org/getting-started/overview)
