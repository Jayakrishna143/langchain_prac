from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Or gemini-1.5-flash depending on your API access
    temperature=0,
    google_api_key=os.getenv("gemini")
)

# Strip out the share parameters to leave only the 11-character ID
video_id = "lsPx6BaiPY0"

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Combine the text chunks into one continuous string
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
except Exception as e:
    # Adding a generic exception block helps catch other API issues (like video not found)
    print(f"An error occurred: {e}")