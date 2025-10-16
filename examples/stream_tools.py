import asyncio

from agents import (
    function_tool,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from agents.tool_context import ToolContext
from agents_ext import Agent, EventQueue, StreamToolContext, run_streamed
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent


@function_tool
async def query_weather(ctx: ToolContext[EventQueue], location: str):
    sctx = StreamToolContext(ctx)

    try:
        sctx.put_reasoning_item("Starting...")
        return f"Weather in {location} is sunny."
    finally:
        sctx.put_reasoning_item("Completed!")


agent = Agent(
    name="weather_agent",
    instructions="You are a helpful weather reporter.",
    tools=[query_weather],
)


async def main():
    result = run_streamed(
        agent, "What's the weather like in Beijing?", context=EventQueue()
    )
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
        elif isinstance(event, RunItemStreamEvent):
            if event.item.type == "tool_call_item":
                print(f"\n['{event.item.raw_item.name}' tool called]", flush=True)
            elif event.item.type == "tool_call_output_item":
                print(f"\n[tool output: {event.item.output}]", flush=True)
            elif event.item.type == "reasoning_item":
                print(
                    f"\n['{event.item.raw_item.id}' reasoning: {event.item.raw_item.summary[0].text}]",
                    flush=True,
                )


if __name__ == "__main__":
    asyncio.run(main())
