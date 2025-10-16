import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from agents import (
    Agent as RawAgent,
    function_tool,
    RunContextWrapper,
    ReasoningItem,
    RunResultStreaming,
)
from agents.agent import _transforms, MaybeAwaitable, AgentBase, TContext, Tool
from agents.items import TResponseInputItem
from agents.lifecycle import RunHooks
from agents.memory.session import Session
from agents.result import RunResultBase
from agents.run import Runner, RunConfig, QueueCompleteSentinel
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    StreamEvent,
)
from agents.tool_context import ToolContext
from openai.types.responses import ResponseReasoningItem, ResponseTextDeltaEvent
from openai.types.responses.response_reasoning_item import Summary


class Agent(RawAgent[TContext]):
    def as_stream_tool(
        self,
        tool_name: str | None,
        tool_description: str | None,
        is_enabled: bool
        | Callable[
            [RunContextWrapper[Any], AgentBase[Any]], MaybeAwaitable[bool]
        ] = True,
        run_config: RunConfig | None = None,
        max_turns: int | None = None,
        hooks: RunHooks[TContext] | None = None,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        session: Session | None = None,
    ) -> Tool:
        """Transform this agent into a streaming tool, callable by other agents.

        This is different from handoffs in two ways:
        1. In handoffs, the new agent receives the conversation history. In this tool, the new agent
           receives generated input.
        2. In handoffs, the new agent takes over the conversation. In this tool, the new agent is
           called as a tool, and the conversation is continued by the original agent.

        This is different from the `as_tool` method in that it streams the events from the agent
        back to the caller. This is useful for long-running tasks where you want to provide feedback
        to the user as the task progresses.

        Args:
            tool_name: The name of the tool. If not provided, the agent's name will be used.
            tool_description: The description of the tool, which should indicate what it does and
                when to use it.
            is_enabled: Whether the tool is enabled. Can be a bool or a callable that takes the run
                context and agent and returns whether the tool is enabled. Disabled tools are hidden
                from the LLM at runtime.
        """

        @function_tool(
            name_override=tool_name
            or _transforms.transform_string_function_style(self.name),
            description_override=tool_description or "",
            is_enabled=is_enabled,
        )
        async def run_agent(context: RunContextWrapper, input: str) -> str:
            from agents.run import DEFAULT_MAX_TURNS, Runner

            resolved_max_turns = (
                max_turns if max_turns is not None else DEFAULT_MAX_TURNS
            )

            result: RunResultStreaming = Runner.run_streamed(
                starting_agent=self,
                input=input,
                context=context,
                max_turns=resolved_max_turns,
                hooks=hooks,
                run_config=run_config,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                session=session,
            )

            tool_output = ""

            queue = StreamToolContext(context).queue

            async for event in result.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        tool_output += event.data.delta
                elif isinstance(event, RunItemStreamEvent):
                    if queue:
                        queue.put_nowait(event)

            return tool_output

        return run_agent


@dataclass
class EventQueue:
    queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = field(
        default_factory=asyncio.Queue,
    )


class StreamToolContext:
    def __init__(self, ctx: ToolContext[EventQueue]) -> None:
        self.ctx = ctx
        self._tree = []

    @property
    def tree(self) -> list[dict]:
        if not hasattr(self, "_tree_calculated"):
            ctx = self.ctx
            while True:
                if not isinstance(ctx, ToolContext):
                    break
                # Child tool context comes first, so we need to reverse the order.
                self._tree.insert(0, ctx)
                ctx = ctx.context
            self._tree_calculated = True
        return self._tree

    @property
    def queue(self) -> asyncio.Queue | None:
        if not self.tree:
            return None

        # The root tool context contains the queue.
        ctx = self.tree[0].context
        if isinstance(ctx, EventQueue):
            return ctx.queue

        return None

    def put_reasoning_item(self, text: str) -> None:
        item = ReasoningItem(
            type="reasoning_item",
            agent="assistant",
            raw_item=ResponseReasoningItem(
                type="reasoning",
                id="/".join(ctx.tool_name for ctx in self.tree),
                summary=[Summary(type="summary_text", text=text)],
            ),
        )
        event = RunItemStreamEvent(
            type="run_item_stream_event",
            name="reasoning_item_created",
            item=item,
        )
        if self.queue:
            self.queue.put_nowait(event)


class StreamResult:
    def __init__(self, raw_result: RunResultStreaming, queue: asyncio.Queue) -> None:
        self.raw_result: RunResultStreaming = raw_result

        self._queue: asyncio.Queue = queue
        self._task: asyncio.Task = asyncio.create_task(self._consume_result())

    async def _consume_result(self) -> None:
        async for event in self.raw_result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                self._queue.put_nowait(event)
            elif isinstance(event, RunItemStreamEvent):
                self._queue.put_nowait(event)

        # Signal that the queue is complete and can be closed.
        self._queue.put_nowait(QueueCompleteSentinel())

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Stream deltas for the internal event queue."""
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break

            if isinstance(item, QueueCompleteSentinel):
                self._queue.task_done()
                break

            yield item
            self._queue.task_done()


def run_streamed(
    starting_agent: Agent[TContext],
    input: str | list[TResponseInputItem],
    context: EventQueue,
    max_turns: int = 10,
    hooks: RunHooks[TContext] | None = None,
    run_config: RunConfig | None = None,
    previous_response_id: str | None = None,
    conversation_id: str | None = None,
    session: Session | None = None,
) -> StreamResult:
    result: RunResultStreaming = Runner.run_streamed(
        starting_agent=starting_agent,
        input=input,
        context=context,
        max_turns=max_turns,
        hooks=hooks,
        run_config=run_config,
        previous_response_id=previous_response_id,
        conversation_id=conversation_id,
        session=session,
    )
    return StreamResult(result, context.queue)


async def run_demo_loop(
    agent: Agent[Any], *, stream: bool = True, context: EventQueue
) -> None:
    """Run a simple REPL loop with the given agent.

    This utility allows quick manual testing and debugging of an agent from the
    command line. Conversation state is preserved across turns. Enter ``exit``
    or ``quit`` to stop the loop.

    Args:
        agent: The starting agent to run.
        stream: Whether to stream the agent output.
        context: Additional context information to pass to the runner.
    """

    current_agent = agent
    input_items: list[TResponseInputItem] = []
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        input_items.append({"role": "user", "content": user_input})

        result: RunResultBase
        if stream:
            stream_result: StreamResult = run_streamed(
                current_agent, input=input_items, context=context
            )
            result = stream_result.raw_result
            async for event in stream_result.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        print(event.data.delta, end="", flush=True)
                elif isinstance(event, RunItemStreamEvent):
                    if event.item.type == "tool_call_item":
                        print(
                            f"\n['{event.item.raw_item.name}' tool called]", flush=True
                        )
                    elif event.item.type == "tool_call_output_item":
                        print(f"\n[tool output: {event.item.output}]", flush=True)
                    elif event.item.type == "reasoning_item":
                        print(
                            f"\n['{event.item.raw_item.id}' reasoning: {event.item.raw_item.summary[0].text}]",
                            flush=True,
                        )
                elif isinstance(event, AgentUpdatedStreamEvent):
                    print(f"\n[Agent updated: {event.new_agent.name}]", flush=True)
            print()
        else:
            result = await Runner.run(current_agent, input_items, context=context)
            if result.final_output is not None:
                print(result.final_output)

        current_agent = result.last_agent
        input_items = result.to_input_list()
