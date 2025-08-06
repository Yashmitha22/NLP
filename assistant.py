import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
import webbrowser
import os
import requests
import json
from typing import Optional, Dict, Any
import threading
import time

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.tts_engine = pyttsx3.init()
        self.setup_tts_voice()
        
        self.is_listening = False
        self.wake_word = "hey assistant"
        
        print("Adjusting for ambient noise... Please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("Ready to listen!")
    
    def setup_tts_voice(self):
        """Configure text-to-speech voice settings"""
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        self.tts_engine.setProperty('rate', 180)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    
    def speak(self, text: str):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for audio input and convert to text"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)
            
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio).lower()
            print(f"You said: {text}")
            return text
        
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
    
    def get_current_time(self) -> str:
        """Get current time"""
        now = datetime.datetime.now()
        return now.strftime("It's currently %I:%M %p on %B %d, %Y")
    
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for information"""
        try:

            summary = wikipedia.summary(query, sentences=2)
            return f"According to Wikipedia: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
         
            try:
                summary = wikipedia.summary(e.options[0], sentences=2)
                return f"According to Wikipedia: {summary}"
            except:
                return f"I found multiple results for {query}. Can you be more specific?"
        except wikipedia.exceptions.PageError:
            return f"I couldn't find any information about {query} on Wikipedia."
        except Exception as e:
            return f"Sorry, I encountered an error while searching: {str(e)}"
    
    def get_weather(self, city: str = "London") -> str:
        """Get weather information (requires API key for full functionality)"""
        return f"I would need a weather API key to get real weather data for {city}. This is a demo response."
    
    def open_website(self, url: str):
        """Open a website in the default browser"""
        webbrowser.open(url)
        return f"Opening {url} in your browser."
    
    def process_command(self, command: str) -> str:
        """Process voice commands and return appropriate responses"""
        command = command.lower().strip()
        
        
        if any(word in command for word in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        
        elif any(word in command for word in ['time', 'clock', 'what time']):
            return self.get_current_time()
        
        elif any(word in command for word in ['date', 'today', 'what day']):
            today = datetime.datetime.now()
            return f"Today is {today.strftime('%A, %B %d, %Y')}"
        
        elif 'wikipedia' in command or 'search for' in command or 'tell me about' in command:
            search_terms = command.replace('wikipedia', '').replace('search for', '').replace('tell me about', '').strip()
            if search_terms:
                return self.search_wikipedia(search_terms)
            else:
                return "What would you like me to search for?"
        
        elif 'weather' in command:
            return self.get_weather()
        
        elif 'open' in command and any(site in command for site in ['google', 'youtube', 'facebook', 'twitter']):
            if 'google' in command:
                self.open_website('https://www.google.com')
                return "Opening Google."
            elif 'youtube' in command:
                self.open_website('https://www.youtube.com')
                return "Opening YouTube."
            elif 'facebook' in command:
                self.open_website('https://www.facebook.com')
                return "Opening Facebook."
            elif 'twitter' in command:
                self.open_website('https://www.twitter.com')
                return "Opening Twitter."
        
        # Calculator
        elif any(word in command for word in ['calculate', 'math', 'plus', 'minus', 'multiply', 'divide']):
            return self.simple_calculator(command)
        
        # Exit commands
        elif any(word in command for word in ['goodbye', 'bye', 'exit', 'quit', 'stop']):
            return "Goodbye! Have a great day!"
        
        # Default response for unrecognized commands
        else:
            return "I'm sorry, I didn't understand that command. Can you please rephrase or ask me about time, weather, Wikipedia searches, or opening websites?"
    
    def simple_calculator(self, command: str) -> str:
        """Simple calculator functionality"""
        try:
            # Extract numbers and operations
            import re
            
            if 'plus' in command or '+' in command:
                numbers = re.findall(r'\d+', command)
                if len(numbers) >= 2:
                    result = int(numbers[0]) + int(numbers[1])
                    return f"{numbers[0]} plus {numbers[1]} equals {result}"
            
            elif 'minus' in command or '-' in command:
                numbers = re.findall(r'\d+', command)
                if len(numbers) >= 2:
                    result = int(numbers[0]) - int(numbers[1])
                    return f"{numbers[0]} minus {numbers[1]} equals {result}"
            
            elif 'multiply' in command or 'times' in command or '*' in command:
                numbers = re.findall(r'\d+', command)
                if len(numbers) >= 2:
                    result = int(numbers[0]) * int(numbers[1])
                    return f"{numbers[0]} times {numbers[1]} equals {result}"
            
            elif 'divide' in command or '/' in command:
                numbers = re.findall(r'\d+', command)
                if len(numbers) >= 2:
                    if int(numbers[1]) != 0:
                        result = int(numbers[0]) / int(numbers[1])
                        return f"{numbers[0]} divided by {numbers[1]} equals {result:.2f}"
                    else:
                        return "Cannot divide by zero!"
            
            return "I couldn't understand the math problem. Try saying something like 'five plus three' or 'ten minus two'."
        
        except Exception as e:
            return "I had trouble with that calculation. Please try again."
    
    def start_listening(self):
        """Main loop to continuously listen for commands"""
        self.speak("Voice assistant activated. Say 'hey assistant' followed by your question or command.")
        
        while True:
            try:
                # Listen for the wake word
                audio_input = self.listen(timeout=1)
                
                if audio_input and self.wake_word in audio_input:
                    self.speak("Yes, how can I help you?")
                    
                    # Listen for the actual command
                    command = self.listen(timeout=5)
                    
                    if command:
                        if any(word in command for word in ['goodbye', 'bye', 'exit', 'quit', 'stop']):
                            self.speak("Goodbye! Have a great day!")
                            break
                        
                        response = self.process_command(command)
                        self.speak(response)
                    else:
                        self.speak("I didn't hear anything. Please try again.")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                self.speak("Voice assistant shutting down. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

def main():
    """Main function to run the voice assistant"""
    print("Initializing Voice Assistant...")
    print("Make sure you have a working microphone and speakers.")
    print("Say 'hey assistant' followed by your question or command.")
    print("Say 'hey assistant goodbye' to exit.")
    print("-" * 50)
    
    try:
        assistant = VoiceAssistant()
        assistant.start_listening()
    except Exception as e:
        print(f"Failed to initialize voice assistant: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install speechrecognition pyttsx3 wikipedia pyaudio")

if __name__ == "__main__":
    main()