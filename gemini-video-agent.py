import logging
import os  # Add this import at the top

from dotenv import load_dotenv
from google.genai import types

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins.google import beta

logger = logging.getLogger("gemini-video-agent")

load_dotenv()


class GeminiAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="you are gemini, a helpful assistant. Read out what is present on the screen.",
            llm=beta.realtime.RealtimeModel(
                input_audio_transcription=types.AudioTranscriptionConfig(),
                vertexai=True,
                project="logical-fort-449407-g9",
                location="us-central1",
            ),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="introduce yourself very briefly and ask about the user's day"
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await ctx.wait_for_participant()

    await session.start(
        agent=GeminiAgent(),
        room=ctx.room,
        # by default, video is disabled
        room_input_options=RoomInputOptions(video_enabled=True),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    # Load environment variables if not already loaded by dotenv
    LIVEKIT_URL = os.getenv("NEXT_PUBLIC_LIVEKIT_URL")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.error("LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set")
        exit(1)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
    )