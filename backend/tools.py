"""
Tool definitions for the QA Agent.
Each tool has a name, description, and an execute function.
"""

import math
import datetime
import json
import re
import httpx

# ──────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────

TOOL_REGISTRY: dict = {}


def register_tool(name: str, description: str):
    """Decorator to register a tool in the global registry."""
    def decorator(func):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "execute": func,
        }
        return func
    return decorator


# ──────────────────────────────────────────────
# 1. Calculator Tool
# ──────────────────────────────────────────────

@register_tool(
    name="calculator",
    description="Evaluates mathematical expressions. Input should be a valid math expression like '2 + 2', 'sqrt(144)', '15 * 3.14', 'sin(45)', 'log(100)'. Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, log10, pi, e, abs, round, ceil, floor."
)
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Clean the expression
        expr = expression.strip()

        # Define safe math functions and constants
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "ceil": math.ceil,
            "floor": math.floor,
            "pow": pow,
            "factorial": math.factorial,
        }

        # Remove potentially dangerous characters
        allowed_chars = set("0123456789+-*/().,%^ ")
        allowed_words = set(safe_dict.keys())

        # Extract words from expression
        words = re.findall(r'[a-zA-Z_]+', expr)
        for word in words:
            if word not in allowed_words:
                return f"Error: Unknown function or variable '{word}'. Available: {', '.join(sorted(allowed_words))}"

        # Replace ^ with ** for exponentiation
        expr = expr.replace("^", "**")

        # Evaluate safely
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        
        # Format result
        if isinstance(result, float):
            if result == int(result):
                return str(int(result))
            return f"{result:.6f}".rstrip("0").rstrip(".")
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# ──────────────────────────────────────────────
# 2. Wikipedia Tool
# ──────────────────────────────────────────────

@register_tool(
    name="wikipedia",
    description="Searches Wikipedia for information about a topic. Input should be a search query like 'Albert Einstein', 'quantum physics', 'Python programming language'. Returns a summary of the most relevant Wikipedia article."
)
def wikipedia_search(query: str) -> str:
    """Fetch a summary from Wikipedia."""
    try:
        # Use Wikipedia's REST API
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.strip().replace(" ", "_")
        
        response = httpx.get(
            url,
            headers={"User-Agent": "QA-Agent/1.0"},
            follow_redirects=True,
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            title = data.get("title", query)
            extract = data.get("extract", "No summary available.")
            
            # Truncate if too long
            if len(extract) > 1000:
                extract = extract[:1000] + "..."
            
            return f"Wikipedia: {title}\n{extract}"
        
        # If direct lookup fails, try search
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query.strip(),
            "format": "json",
            "srlimit": 1,
        }
        
        response = httpx.get(
            search_url,
            params=params,
            headers={"User-Agent": "QA-Agent/1.0"},
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                snippet = re.sub(r'<[^>]+>', '', results[0].get("snippet", ""))
                
                # Now get the full summary
                summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
                summary_resp = httpx.get(
                    summary_url,
                    headers={"User-Agent": "QA-Agent/1.0"},
                    follow_redirects=True,
                    timeout=10.0
                )
                if summary_resp.status_code == 200:
                    summary_data = summary_resp.json()
                    extract = summary_data.get("extract", snippet)
                    if len(extract) > 1000:
                        extract = extract[:1000] + "..."
                    return f"Wikipedia: {title}\n{extract}"
                
                return f"Wikipedia: {title}\n{snippet}"
        
        return f"No Wikipedia results found for '{query}'."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# ──────────────────────────────────────────────
# 3. Web Search Tool (DuckDuckGo)
# ──────────────────────────────────────────────

@register_tool(
    name="web_search",
    description="Searches the web using DuckDuckGo for current information. Input should be a search query like 'latest AI news 2024', 'weather in New York', 'Python 3.12 new features'. Returns top search results with titles and snippets."
)
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query.strip(), max_results=5))

        if not results:
            return f"No web search results found for '{query}'."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No description")
            href = r.get("href", "")
            formatted.append(f"{i}. **{title}**\n   {body}\n   Source: {href}")

        return "Web Search Results:\n\n" + "\n\n".join(formatted)
    except ImportError:
        return "Error: duckduckgo-search package not installed. Install with: pip install duckduckgo-search"
    except Exception as e:
        return f"Error performing web search: {str(e)}"


# ──────────────────────────────────────────────
# 4. DateTime Tool
# ──────────────────────────────────────────────

@register_tool(
    name="datetime",
    description="Gets the current date, time, and timezone information. Input can be: 'now' for current datetime, 'date' for just the date, 'time' for just the time, 'day' for the day of the week, or a timezone like 'UTC', 'US/Eastern', 'Asia/Tokyo'."
)
def datetime_tool(query: str) -> str:
    """Get current date/time information."""
    try:
        now = datetime.datetime.now()
        query = query.strip().lower()

        if query in ("now", "current", "datetime"):
            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} (Local timezone)"
        elif query == "date":
            return f"Current date: {now.strftime('%A, %B %d, %Y')}"
        elif query == "time":
            return f"Current time: {now.strftime('%I:%M:%S %p')}"
        elif query in ("day", "weekday"):
            return f"Today is {now.strftime('%A')}"
        elif query == "year":
            return f"Current year: {now.year}"
        elif query == "month":
            return f"Current month: {now.strftime('%B %Y')}"
        else:
            # Default: return full info
            return (
                f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Day: {now.strftime('%A')}\n"
                f"Date: {now.strftime('%B %d, %Y')}\n"
                f"Time: {now.strftime('%I:%M:%S %p')}\n"
                f"Timezone: Local system timezone"
            )
    except Exception as e:
        return f"Error getting datetime: {str(e)}"


# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────

def get_tool_descriptions() -> str:
    """Generate formatted tool descriptions for the system prompt."""
    descriptions = []
    for name, tool in TOOL_REGISTRY.items():
        descriptions.append(f"- **{name}**: {tool['description']}")
    return "\n".join(descriptions)


def execute_tool(tool_name: str, tool_input: str) -> str:
    """Execute a tool by name with the given input."""
    tool_name = tool_name.strip().lower()
    if tool_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[tool_name]["execute"](tool_input)
    else:
        available = ", ".join(TOOL_REGISTRY.keys())
        return f"Error: Unknown tool '{tool_name}'. Available tools: {available}"


def list_tools() -> list:
    """Return a list of available tools with their info."""
    return [
        {"name": name, "description": tool["description"]}
        for name, tool in TOOL_REGISTRY.items()
    ]
