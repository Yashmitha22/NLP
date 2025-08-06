# Voice Assistant

A Python-based voice assistant that can respond to voice commands and questions using speech recognition and text-to-speech.

## Features

- **Voice Recognition**: Listens for the wake word "hey assistant" followed by commands
- **Text-to-Speech**: Responds with voice output
- **Multiple Capabilities**:
  - Tell current time and date
  - Search Wikipedia for information
  - Simple math calculations
  - Open popular websites (Google, YouTube, Facebook, Twitter)
  - Weather queries (placeholder - requires API key)
  - Greeting and conversation

## Installation

### Method 1: Using setup script (Recommended)
```bash
python setup.py
```

### Method 2: Manual installation
```bash
pip install -r requirements.txt
```

### Method 3: Individual packages
```bash
pip install speechrecognition pyttsx3 wikipedia requests
pip install pyaudio
```

**Note**: PyAudio can be tricky to install on Windows. If it fails, try:
```bash
pip install pipwin
pipwin install pyaudio
```

## Usage

1. Run the assistant:
```bash
python assistant.py
```

2. Wait for the message "Ready to listen!"

3. Say the wake word "hey assistant" followed by your command

### Example Commands

- "Hey assistant, what time is it?"
- "Hey assistant, tell me about Python programming"
- "Hey assistant, what's five plus three?"
- "Hey assistant, open Google"
- "Hey assistant, what's the date today?"
- "Hey assistant, goodbye"

## Supported Commands

| Category | Examples |
|----------|----------|
| **Time/Date** | "what time is it?", "what's the date?" |
| **Wikipedia** | "tell me about [topic]", "search for [topic]" |
| **Math** | "five plus three", "ten minus two", "multiply 4 and 6" |
| **Websites** | "open Google", "open YouTube" |
| **Greetings** | "hello", "hi" |
| **Exit** | "goodbye", "bye", "exit", "stop" |

## Requirements

- Python 3.6+
- Working microphone
- Speakers or headphones
- Internet connection (for speech recognition and Wikipedia)

## Troubleshooting

### Common Issues

1. **Microphone not working**: Check your microphone permissions and make sure it's set as default
2. **PyAudio installation fails**: See installation notes above
3. **Speech recognition errors**: Ensure you have a stable internet connection
4. **No audio output**: Check your speaker/headphone connections

### Dependencies Issues
If you encounter issues with dependencies, try upgrading pip first:
```bash
python -m pip install --upgrade pip
```

## Customization

You can customize the assistant by modifying `assistant.py`:

- Change the wake word by modifying `self.wake_word`
- Add new commands in the `process_command()` method
- Adjust speech rate and volume in `setup_tts_voice()`
- Add new features by creating additional methods

## API Integration

To add real weather data, you'll need to:
1. Sign up for a weather API (like OpenWeatherMap)
2. Add your API key to the `get_weather()` method
3. Implement the API call

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the voice assistant!
