"""
Agent Core — ReAct (Reasoning + Acting) pattern implementation.
Uses Ollama with Llama3 for reasoning and tool execution.
"""

import re
import ollama
from typing import Generator

from backend.config import (
    OLLAMA_MODEL,
    MAX_AGENT_ITERATIONS,
    AGENT_TEMPERATURE,
    SYSTEM_PROMPT,
)
from backend.tools import get_tool_descriptions, execute_tool


class AgentStep:
    """Represents a single step in the agent's reasoning chain."""
    def __init__(self, step_type: str, content: str):
        self.step_type = step_type  # 'thought', 'action', 'action_input', 'observation', 'final_answer'
        self.content = content

    def to_dict(self):
        return {"type": self.step_type, "content": self.content}


class QAAgent:
    """
    Question Answering Agent using ReAct pattern with Ollama/Llama3.
    
    The agent reasons step-by-step, decides when to use tools,
    and produces a final answer after gathering enough information.
    """

    def __init__(self):
        self.conversation_history: list[dict] = []
        self.model = OLLAMA_MODEL
        self.system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=get_tool_descriptions()
        )

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def _parse_agent_response(self, text: str) -> list[AgentStep]:
        """
        Parse the agent's response to extract Thought, Action, Action Input, and Final Answer.
        
        IMPORTANT: We stop parsing after Action Input is found, ignoring any
        model-hallucinated Observation or Final Answer that follows a tool call.
        The real Observation will be injected by the agent loop.
        """
        steps = []
        lines = text.strip().split("\n")
        
        current_type = None
        current_content = []
        found_action_input = False

        for line in lines:
            line_stripped = line.strip()
            
            # If we already found an action_input, stop parsing.
            # The model may hallucinate Observation/Final Answer after a tool call,
            # but we need to inject the REAL tool output.
            if found_action_input and current_type == "action_input":
                # Check if this line is a new marker — if so, we stop
                if any(line_stripped.startswith(m) for m in 
                       ["Observation:", "Final Answer:", "Thought:", "Action:", "Action Input:"]):
                    # Save the action_input we have and stop
                    if current_content:
                        steps.append(AgentStep(current_type, "\n".join(current_content).strip()))
                    break
                else:
                    # Continue accumulating action_input content (multi-line input)
                    current_content.append(line_stripped)
                    continue

            # Check for markers
            if line_stripped.startswith("Thought:"):
                if current_type and current_content:
                    steps.append(AgentStep(current_type, "\n".join(current_content).strip()))
                current_type = "thought"
                current_content = [line_stripped[len("Thought:"):].strip()]
            elif line_stripped.startswith("Action:"):
                if current_type and current_content:
                    steps.append(AgentStep(current_type, "\n".join(current_content).strip()))
                current_type = "action"
                current_content = [line_stripped[len("Action:"):].strip()]
            elif line_stripped.startswith("Action Input:"):
                if current_type and current_content:
                    steps.append(AgentStep(current_type, "\n".join(current_content).strip()))
                current_type = "action_input"
                current_content = [line_stripped[len("Action Input:"):].strip()]
                found_action_input = True
            elif line_stripped.startswith("Final Answer:"):
                if current_type and current_content:
                    steps.append(AgentStep(current_type, "\n".join(current_content).strip()))
                current_type = "final_answer"
                current_content = [line_stripped[len("Final Answer:"):].strip()]
            elif line_stripped.startswith("Observation:"):
                # Only accept Observation if we haven't found an action_input
                # (i.e., this is from a scratchpad continuation, not hallucinated)
                if current_type and current_content:
                    steps.append(AgentStep(current_type, "\n".join(current_content).strip()))
                current_type = "observation"
                current_content = [line_stripped[len("Observation:"):].strip()]
            else:
                if current_type:
                    current_content.append(line_stripped)

        # Don't forget the last block
        if current_type and current_content:
            steps.append(AgentStep(current_type, "\n".join(current_content).strip()))

        return steps

    def _build_messages(self, scratchpad: str = "") -> list[dict]:
        """Build the message list for the Ollama API call."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history (previous turns)
        messages.extend(self.conversation_history)
        
        # If there's a scratchpad (current reasoning chain), append it
        if scratchpad:
            messages.append({"role": "assistant", "content": scratchpad})
        
        return messages

    def _truncate_response_at_action(self, text: str) -> str:
        """
        Truncate the model response after Action Input line.
        This prevents hallucinated observations from being included in the scratchpad.
        """
        lines = text.strip().split("\n")
        result_lines = []
        found_action_input = False
        
        for line in lines:
            result_lines.append(line)
            if line.strip().startswith("Action Input:"):
                found_action_input = True
            elif found_action_input:
                # If we've passed Action Input and hit a new marker, stop
                if any(line.strip().startswith(m) for m in 
                       ["Observation:", "Final Answer:", "Thought:"]):
                    result_lines.pop()  # Remove this hallucinated line
                    break
        
        return "\n".join(result_lines)

    def run(self, user_question: str) -> Generator[AgentStep, None, None]:
        """
        Run the agent on a user question.
        Yields AgentStep objects as the agent reasons through the problem.
        """
        # Add user question to history
        self.conversation_history.append({
            "role": "user",
            "content": user_question
        })

        scratchpad = ""
        final_answer = None

        for iteration in range(MAX_AGENT_ITERATIONS):
            # Build messages
            messages = self._build_messages(scratchpad if scratchpad else "")

            try:
                # Call Ollama
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": AGENT_TEMPERATURE,
                        "num_predict": 1024,
                        "stop": ["Observation:"],  # Stop generation before hallucinating observations
                    }
                )
                
                response_text = response["message"]["content"]
            except Exception as e:
                yield AgentStep("error", f"Error calling Ollama: {str(e)}")
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}"
                })
                return

            # Parse the response
            steps = self._parse_agent_response(response_text)

            if not steps:
                # If no structured output, treat the whole response as a final answer
                yield AgentStep("final_answer", response_text)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                return

            # Process each step
            action_name = None
            action_input = None

            for step in steps:
                # Don't yield hallucinated observations
                if step.step_type == "observation":
                    continue
                    
                yield step

                if step.step_type == "final_answer":
                    final_answer = step.content
                    break
                elif step.step_type == "action":
                    action_name = step.content
                elif step.step_type == "action_input":
                    action_input = step.content

            # If we got a final answer, we're done
            if final_answer:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_answer
                })
                return

            # If we have an action to execute
            if action_name and action_input is not None:
                # Execute the tool
                observation = execute_tool(action_name, action_input)
                
                # Yield the REAL observation
                obs_step = AgentStep("observation", observation)
                yield obs_step

                # Truncate response to remove any hallucinated content after Action Input
                clean_response = self._truncate_response_at_action(response_text)
                
                # Add to scratchpad for next iteration
                scratchpad += clean_response + f"\nObservation: {observation}\n"
            else:
                # No action and no final answer — treat response as final answer
                full_text = "\n".join(s.content for s in steps if s.step_type != "observation")
                if not full_text.strip():
                    full_text = response_text
                yield AgentStep("final_answer", full_text)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_text
                })
                return

        # Max iterations reached
        if scratchpad:
            yield AgentStep(
                "final_answer",
                "I've been thinking about this for a while. Based on what I've gathered so far, "
                "let me give you the best answer I can with the information available."
            )
            self.conversation_history.append({
                "role": "assistant",
                "content": "I reached the maximum reasoning steps. Here's what I found based on available information."
            })

    def get_history(self) -> list[dict]:
        """Return conversation history."""
        return self.conversation_history
