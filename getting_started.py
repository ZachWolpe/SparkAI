import asyncio
import random

import numpy as np
import sounddevice as sd
from agents import Agent, function_tool  # set_tracing_disabled,
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

language = "English"


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


language_agent = Agent(
    name=language,
    handoff_description=f"A {language} speaking agent.",
    instructions=prompt_with_handoff_instructions(
        f"You're speaking to a human, so be polite and concise. Speak in {language}.",
    ),
    model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        f"You're speaking to a human, so be polite and concise. If the user speaks in {language}, handoff to the {language} agent.",
    ),
    model="gpt-4o-mini",
    handoffs=[language_agent],
    tools=[get_weather],
)


async def main():
    print("launching agent....")
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    buffer = np.zeros(24000 * 3, dtype=np.int16)
    audio_input = AudioInput(buffer=buffer)
    print("audio_input: ")
    print(audio_input)

    result = await pipeline.run(audio_input)
    print("result: ")
    print(result)

    # Create an audio player using `sounddevice`
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    # Play the audio stream as it comes in
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            player.write(event.data)


if __name__ == "__main__":
    asyncio.run(main())
