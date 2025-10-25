# OpenAI Agents Extension

An extension library for the [OpenAI Agents SDK][1], providing enhanced streaming capabilities.


## Features

- **Streaming Events for Tools**: Allow tools to stream their intermediate events back to the agent runtime.
- **Streaming Events for Agents as Tools**: Allow agents to be used as tools within other agents, with streaming events support.
- **Seamless Integration**: Works alongside existing OpenAI Agents SDK functionality.


## Installation

```bash
git clone https://github.com/RussellLuo/openai-agents-ext.git
cd openai-agents-ext
uv pip install -e .
```


## Examples

(Check out the [examples](./examples) directory for runnable examples.)

### Streaming Events for Tools

```python
# stream_tools.py

import asyncio

from agents import function_tool, RawResponsesStreamEvent, RunItemStreamEvent
from agents.tool_context import ToolContext
from agents_ext import Agent, EventQueue, StreamToolContext, run_streamed
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent


@function_tool
async def query_weather(ctx: ToolContext[EventQueue], location: str):
    sctx = StreamToolContext(ctx)

    try:
        sctx.report_progress("Starting...")
        return f"Weather in {location} is sunny."
    finally:
        sctx.report_progress("Completed!")


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

    # I'll check the weather in Beijing for you.
    # ['query_weather' tool called]
    #
    # ['query_weather' reasoning: Starting...]
    #
    # ['query_weather' reasoning: Completed!]
    #
    # [tool output: Weather in Beijing is sunny.]
    # The weather in Beijing is currently sunny!


if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming Events for Agents as Tools

```python
# agents_as_stream_tools.py

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

    # I'll check the weather in Beijing for you.
    # ['weather_agent' tool called]
    # 
    # ['weather_agent/query_weather' tool called]
    # 
    # ['weather_agent/query_weather' reasoning: Starting...]
    # 
    # ['weather_agent/query_weather' reasoning: Completed!]
    # 
    # [tool output: Weather in Beijing is sunny.]
    # 
    # [tool output: I'll get the current weather information for Beijing for you.The current weather in Beijing is **sunny**. It looks like a beautiful day there!
    # The weather in Beijing is currently **sunny**. It looks like a beautiful day there!


if __name__ == "__main__":
    asyncio.run(main())
```


## License

[MIT][2]


[1]: https://github.com/openai/openai-agents-python
[2]: http://opensource.org/licenses/MIT