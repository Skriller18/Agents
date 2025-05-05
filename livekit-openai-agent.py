#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import base64
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, Agent, AgentSession
from livekit.rtc import Track, TrackKind, VideoStream
from livekit.agents import llm, tts
from livekit.plugins.openai import tts as openai_tts
# Import the OpenAI LLM plugin correctly
from livekit.plugins.openai import llm as openai_llm

# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------

# Load variables from .env if present and allow shell exports to override
load_dotenv(override=False)

LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "wss://elemento-u3zedxjs.livekit.cloud")
LIVEKIT_API_KEY: Optional[str] = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET: Optional[str] = os.getenv("LIVEKIT_API_SECRET")

SPEAKING_FRAME_RATE = 1.0
NOT_SPEAKING_FRAME_RATE = 0.5

# System prompt for the math tutor
_SYSTEM_PROMPT = """
You are an educational AI tutor specializing in Math and Physics. You analyze the student's work displayed on the whiteboard/notepad and provide both feedback on their current work and hints to help them move ahead with the given question.

When analyzing the student's work:
1. Check for mathematical correctness, logical flow, units consistency, formula application, and conceptual clarity.
2. Assess the relationship between the question and the text/images the student has written.
3. When multiple mistakes exist, focus on identifying and correcting the FIRST mistake only to avoid overwhelming the student.
4. Provide concise feedback that identifies the specific error with precise details.
5. If everything is correct, provide positive reinforcement. If the student has reached the final solution and it is correct, congratulate them!
6. If the student hasn't reached the final solution, provide a hint that prompts them to think deeply and try again.

Maintain a supportive, friendly tone throughout all responses - like an encouraging tutor who genuinely cares about the student's progress.

If the student provides no work or asks a question directly:
- Provide a concise hint (under 30 words) with specific guidance to help them get to the next step without revealing the complete answer.
- Respond with a friendly, encouraging message that motivates the student to try the problem.
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("math-tutor-agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------


class MathTutorAgent:
    """LiveKit agent that joins a room and tutors students in math & physics."""

    def __init__(self):
        self.agent: Optional[Agent] = None
        self.agent_session: Optional[AgentSession] = None
        self._is_user_speaking: bool = False
        self.room = None  # will hold the Room instance once connected
        self.last_video_frame = None  # Last video frame processed
        self.last_image_data = None  # Last image data from RPC
        self.last_student_question = ""  # Last question received from student
        self.analysis_in_progress = False  # Flag to prevent overlapping analysis

    # ------------------------- LiveKit lifecycle ------------------------- #

    async def start(self, ctx: JobContext):
        """Entrypoint called by livekit‑agents worker."""

        logger.info(f"Starting Math Tutor Agent")
        logger.info(f"Connecting to LiveKit at {LIVEKIT_URL}")

        # ------------------------------------------------------------------
        # Connect to the LiveKit server that issued this job
        # ------------------------------------------------------------------
        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
            logger.info("Connected to LiveKit")
        except Exception:
            logger.error("Failed to connect to LiveKit", exc_info=True)
            raise

        # announce ourselves as an agent so the front‑end can recognise us
        if hasattr(ctx.room.local_participant, "update_metadata"):
                try:
                    await ctx.room.local_participant.update_metadata(json.dumps({"kind": "agent"}))
                except Exception:
                    logger.warning("Could not set participant metadata; continuing without it")
        self.room = ctx.room

        try:
            logger.info("Waiting for participant …")
            participant = await ctx.wait_for_participant()
            logger.info(f"Participant joined: {participant.identity}")

            # Create a chat context first
            chat_ctx = llm.ChatContext.empty()

            # Initialize the agent with instructions
            self.agent = Agent(instructions=_SYSTEM_PROMPT, chat_ctx=chat_ctx)

            # Create a TTS model or use a DummyTTS if you don't need voice
            try:
                tts_model = openai_tts.TTS()
                logger.info("Using OpenAI TTS")
            except Exception as e:
                logger.warning(f"OpenAI TTS not available: {e}, using DummyTTS")
                tts_model = tts.DummyTTS()

            # Create an agent session with LLM and TTS
            # Use the OpenAI LLM from the correct module
            self.agent_session = AgentSession(
                llm=openai_llm.LLM(model="gpt-4o"),
                tts=tts_model,
                allow_interruptions=True,
            )

            # Register RPC & event handlers
            ctx.room.local_participant.register_rpc_method("analyzeImage", self._handle_image_analysis)
            
            # Register event handlers - the correct way
            ctx.room.on("track_subscribed", self._on_track_subscribed)
            ctx.room.on("data_received", self._on_data_received)

            # Start the agent session with our agent in the room
            logger.info("Starting agent session…")
            await self.agent_session.start(agent=self.agent, room=ctx.room)

            # Send a greeting
            await self._send_greeting()

        except Exception:
            logger.error("Error during agent startup", exc_info=True)
            raise

    # ------------------------- Data helpers --------------------------- #

    async def _publish_text(self, text: str):
        """Send a small JSON packet on topic='transcription' so the UI can display it."""
        if not self.room:
            return
        try:
            await self.room.local_participant.publish_data(
                json.dumps({"text": text}).encode("utf-8"), topic="transcription"
            )
            logger.info(f"Published text message: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to publish text data: {e}")

    async def _generate_analysis(self, prompt: str):
        """Generate analysis with the agent and speak/publish the response."""
        if self.analysis_in_progress:
            logger.info("Analysis already in progress, skipping...")
            return
            
        try:
            self.analysis_in_progress = True
            logger.info(f"Generating analysis for prompt: {prompt[:50]}...")
            
            # Generate a response
            reply = await self.agent_session.generate_reply(user_input=prompt)
            
            # Extract the text content
            if hasattr(reply, 'content'):
                reply_text = " ".join(reply.content) if isinstance(reply.content, list) else str(reply.content)
            else:
                reply_text = str(reply)
                
            # Publish the text to the frontend
            await self._publish_text(reply_text)
            
            # Speak the response
            if self.agent_session and hasattr(self.agent_session, 'say'):
                self.agent_session.say(reply_text)
            
            logger.info(f"Generated and published analysis: {reply_text[:50]}...")
            return reply_text
        except Exception as e:
            logger.error(f"Error generating analysis: {e}", exc_info=True)
            await self._publish_text(f"I'm having trouble analyzing your work. Please try again.")
            return None
        finally:
            self.analysis_in_progress = False

    # ------------------------- Event handlers --------------------------- #

    def _on_track_subscribed(self, track: Track, publication, participant):
        """Handle track subscribed event with correct signature."""
        if track.kind == TrackKind.KIND_VIDEO:
            asyncio.create_task(self._handle_video_track(track))
        elif track.kind == TrackKind.KIND_AUDIO:
            # Process audio track - LiveKit handles this automatically for voice chat
            logger.info(f"Received audio track: {track.sid}")

    def _on_data_received(self, data, participant, topic):
        """Handle data received event with correct signature."""
        # Only process data on the transcription topic - this is for user speech
        if topic == "transcription":
            asyncio.create_task(self._handle_data_received(data, participant, topic))

    async def _handle_video_track(self, track: Track):
        """Handle video track from screen share."""
        logger.info(f"Received video track: {track.sid}")
        video_stream = VideoStream(track)
        last_frame_time = 0.0
        frame_counter = 0
        
        try:
            async for frame in video_stream:
                now = asyncio.get_event_loop().time()
                if now - last_frame_time < self._frame_interval():
                    continue
                last_frame_time = now
                frame_counter += 1
                self.last_video_frame = frame

                # Every 30th frame, generate a response about the work
                if frame_counter % 30 == 0 and not self.analysis_in_progress:
                    logger.info(f"Processing video frame {frame_counter}")
                    
                    # Extract image from frame if needed
                    # This depends on what you want to do with the video frames
                    
                    # Only analyze if there's a recent student question or it's been a while
                    if self.last_student_question or frame_counter % 120 == 0:
                        prompt = self.last_student_question or "Please analyze what you see on my canvas."
                        self.last_student_question = ""  # Clear after using
                        await self._generate_analysis(prompt)
                        
        except Exception as e:
            logger.error(f"Error handling video stream: {e}", exc_info=True)
        finally:
            await video_stream.aclose()
            logger.info("Video stream closed")

    async def _handle_data_received(self, payload, participant, topic):
        """Handle data packets received on the transcription topic."""
        if topic != "transcription":
            return
            
        try:
            message = json.loads(payload.decode("utf-8"))
            if "text" in message:
                text = message["text"]
                logger.info(f"Received transcription: {text}")
                
                # Set as the latest question
                self.last_student_question = text
                
                # Generate a response
                await self._generate_analysis(text)
                
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)

    async def _handle_image_analysis(self, request_data: str) -> str:
        """Handle image analysis RPC call from frontend."""
        try:
            data = json.loads(request_data)
            if "imageData" not in data:
                return json.dumps({"error": "No image data provided"})
                
            image_data = data["imageData"]
            self.last_image_data = image_data
            
            # Only process if we haven't recently analyzed an image
            if not self.analysis_in_progress:
                # Convert the image data to an actual image if needed
                # image_base64 = image_data.split(',')[1] if ',' in image_data else image_data
                # self._process_image(image_base64)
                
                # Generate analysis with current context from the student's canvas
                prompt = self.last_student_question or "Please analyze what I've written on the canvas."
                self.last_student_question = ""  # Clear after using
                
                await self._generate_analysis(prompt)
                
            return json.dumps({"status": "success", "message": "Image analysis complete"})
        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            return json.dumps({"error": str(e)})

    def _process_image(self, image_base64: str) -> None:
        """Process an image from base64 string (optional)."""
        try:
            # Convert base64 to image
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # Here you could perform additional image processing
            # For example, OCR to extract text from the image
            
            logger.info(f"Processed image of size: {image.size}")
        except Exception as e:
            logger.error(f"Error processing image: {e}")

    # ------------------------- Helpers --------------------------- #

    async def _send_greeting(self):
        """Send a greeting when the agent starts."""
        greeting = (
            "Hello! I'm your Math and Physics tutor. I can see your work and help guide you through problems. "
            "Draw or write on the canvas, and I'll provide feedback and hints to help you succeed!"
        )
        
        # Send to browser
        await self._publish_text(greeting)
        
        # Speak the greeting
        if self.agent_session and hasattr(self.agent_session, 'say'):
            self.agent_session.say(greeting)

    def _frame_interval(self) -> float:
        """Calculate frame interval based on speaking state."""
        return 1.0 / (SPEAKING_FRAME_RATE if self._is_user_speaking else NOT_SPEAKING_FRAME_RATE)


# ---------------------------------------------------------------------------
# Entrypoint for LiveKit worker
# ---------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit worker."""
    try:
        agent = MathTutorAgent()
        await agent.start(ctx)
        
        # Keep agent running
        while True:
            await asyncio.sleep(60)
            logger.info("Agent heartbeat – still running")
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    from livekit.agents import cli, WorkerOptions

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
    )