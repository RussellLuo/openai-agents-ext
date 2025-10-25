import asyncio

from agents import RawResponsesStreamEvent, RunItemStreamEvent
from agents_ext import Agent, EventQueue, run_streamed
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from stream_tools import agent as weather_agent  # noqa: F401


orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions="You are a helpful assistant.",
    tools=[
        weather_agent.as_stream_tool(
            tool_name="weather_agent",
            tool_description="Query the weather in a given location.",
        ),
    ],
)


async def main():
    result = run_streamed(
        orchestrator_agent, "What's the weather like in Beijing?", context=EventQueue()
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
