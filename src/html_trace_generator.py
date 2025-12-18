"""
HTML Trace Generator for SlideVQA Agent Logs.

Generates a beautiful, interactive Chat UI visualization of agent execution traces.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def generate_html_trace(log_path: Path | str, output_path: Path | str | None = None) -> str:
    """
    Generate an HTML trace from an agent_log.jsonl file.
    
    Args:
        log_path: Path to agent_log.jsonl
        output_path: Optional output path for HTML file. If None, saves next to log file.
    
    Returns:
        Path to generated HTML file
    """
    log_path = Path(log_path)
    
    if output_path is None:
        output_path = log_path.parent / "trace.html"
    else:
        output_path = Path(output_path)
    
    # Read log entries
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Extract metadata
    config = {}
    question_data = {}
    system_prompt = ""
    
    for entry in entries:
        if entry.get("title") == "Configuration":
            config = entry.get("content", {})
        elif entry.get("title") == "Question & Answer":
            question_data = entry.get("content", {})
        elif entry.get("title") == "System Prompt":
            system_prompt = entry.get("content", "")
    
    # Calculate timing
    start_time = None
    end_time = None
    for entry in entries:
        ts = entry.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                if start_time is None or dt < start_time:
                    start_time = dt
                if end_time is None or dt > end_time:
                    end_time = dt
            except ValueError:
                continue
    
    duration = (end_time - start_time).total_seconds() if start_time and end_time else 0
    
    # Generate HTML
    html = generate_html_content(
        entries=entries,
        config=config,
        question_data=question_data,
        system_prompt=system_prompt,
        start_time=start_time,
        duration=duration,
        log_path=log_path,
    )
    
    # Write HTML file
    with open(output_path, "w") as f:
        f.write(html)
    
    return str(output_path)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def format_json_html(data: dict | list | str, indent: int = 2) -> str:
    """Format JSON data as syntax-highlighted HTML."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return f'<pre class="json-block">{escape_html(data)}</pre>'
    
    json_str = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    return f'<pre class="json-block">{escape_html(json_str)}</pre>'


def render_markdown(text: str) -> str:
    """Convert simple markdown to HTML."""
    import re
    
    # Escape HTML first
    html = escape_html(text)
    
    # Convert **bold** to <strong>
    html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
    
    # Convert bullet points (lines starting with *)
    lines = html.split('\n')
    in_list = False
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if this is a bullet point
        if stripped.startswith('* '):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            # Remove the "* " and wrap in <li>
            result_lines.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            # Regular line - preserve with <br> if not empty
            if stripped:
                result_lines.append(line)
            else:
                result_lines.append('<br>')
    
    # Close list if still open
    if in_list:
        result_lines.append('</ul>')
    
    return '\n'.join(result_lines)


def get_entry_icon(level: str) -> str:
    """Get an emoji icon for the log level."""
    icons = {
        "SYSTEM": "⚙️",
        "DATA": "📊",
        "MESSAGE": "💬",
        "AGENT_THINK": "💭",
        "TOOL_CALL": "🛠️",
        "TOOL_RESULT": "📥",
        "DECISION": "🤔",
        "PLAN":  "📋",
        "ERROR": "❌",
    }
    return icons.get(level, "📝")


def render_list_tools_result(content: dict) -> str:
    """Render list_tools result as a styled list."""
    tools = content.get("tools", [])
    
    html_parts = ['<div class="tools-list">']
    for tool in tools:
        name = escape_html(tool.get("name", ""))
        desc = escape_html(tool.get("description", ""))
        best_for = escape_html(tool.get("best_for", ""))
        params = escape_html(tool.get("params", ""))
        
        html_parts.append(f'''
        <div class="tool-item">
            <div class="tool-item-header">
                <span class="tool-item-name">{name}</span>
            </div>
            <div class="tool-item-body">
                <div class="tool-item-desc">{desc}</div>
                {f'<div class="tool-item-meta"><strong>Best for:</strong> {best_for}</div>' if best_for else ''}
                {f'<div class="tool-item-meta"><strong>Params:</strong> {params}</div>' if params else ''}
            </div>
        </div>
        ''')
    
    html_parts.append('</div>')
    return "".join(html_parts)


def render_retrieve_and_rerank_result(content: dict, evidence_pages: list[int], base_path: Path) -> str:
    """Render retrieve_and_rerank result as an image carousel with captions."""
    results = content.get("results", [])
    
    # Build query info header
    query_info_parts = []
    if content.get("method"):
        method_display = content["method"].replace("nvembed", "neural")
        query_info_parts.append(f'<span class="query-param"><strong>Method:</strong> {escape_html(method_display)}</span>')
    if content.get("custom_query"):
        query_info_parts.append(f'<span class="query-param"><strong>Query:</strong> "{escape_html(content["custom_query"])}"</span>')
    if content.get("question"):
        query_info_parts.append(f'<span class="query-param"><strong>Question:</strong> {escape_html(content["question"][:100])}</span>')
    if content.get("num_results") is not None or content.get("num_output") is not None:
        num = content.get("num_results") or content.get("num_output") or len(results)
        query_info_parts.append(f'<span class="query-param"><strong>Results:</strong> {num}</span>')
    if content.get("total_ms"):
        query_info_parts.append(f'<span class="query-param"><strong>Time:</strong> {content["total_ms"]:.0f}ms</span>')
    
    if not results:
        header_html = ' | '.join(query_info_parts) if query_info_parts else ''
        return f'<div class="retrieval-header">{header_html}</div><p>No results found.</p>'
    
    # Determine images base path - find project root (COMSE6998-13)  
    project_root = None
    current = base_path
    while current.parent != current:
        if current.name == "COMSE6998-13":
            project_root = current
            break
        current = current.parent
    
    if project_root is None:
        # Fallback: assume we're already in project, try relative path
        project_root = Path("/home/pg2860/COMSE6998-13")
    
    images_base = project_root / "data" / "slidevqa" / "images"
    
    # Start with query info header
    html_parts = []
    if query_info_parts:
        html_parts.append(f'<div class="retrieval-header">{" | ".join(query_info_parts)}</div>')
    
    html_parts.append('<div class="carousel-container">')
    html_parts.append('<div class="carousel-slides">')
    
    for idx, result in enumerate(results):
        deck = result.get("deck", "")
        slide = result.get("slide", "")
        caption = result.get("caption", "No caption")
        rerank_score = result.get("rerank_score", 0)
        
        # Extract slide number from filename (slide_12_1024.jpg -> 12)
        slide_num = None
        if "_" in slide:
            parts = slide.split("_")
            if len(parts) >= 2:
                try:
                    slide_num = int(parts[1])
                except ValueError:
                    pass
        
        # Check if this is a golden evidence slide
        is_golden = slide_num is not None and slide_num in evidence_pages
        golden_class = " golden-slide" if is_golden else ""
        
        # Construct relative image path
        img_path = images_base / deck / slide
        if img_path.exists():
            # Make path relative to the trace.html location (Q0/ -> query_logs/ -> exp/ -> output/ -> root/)
            rel_path = f"../../../../../data/slidevqa/images/{deck}/{slide}"
        else:
            rel_path = ""
        
        html_parts.append(f'''
        <div class="carousel-slide{golden_class}" data-index="{idx}">
            <div class="slide-header">
                <span class="slide-number">Slide {idx + 1}/{len(results)}</span>
                {f'<span class="golden-badge">🏆 Golden Evidence</span>' if is_golden else ''}
                <span class="slide-score">Score: {rerank_score:.3f}</span>
            </div>
            {f'<img src="{rel_path}" alt="Slide {slide_num}" class="slide-image" loading="lazy" />' if rel_path else '<div class="slide-image-placeholder">Image not found</div>'}
            <div class="slide-caption">{render_markdown(caption)}</div>
            <div class="slide-meta">
                <span class="slide-deck">{escape_html(deck)}</span>
                <span class="slide-file">{escape_html(slide)}</span>
            </div>
        </div>
        ''')
    
    html_parts.append('</div>')  # carousel-slides
    
    # Add navigation buttons
    html_parts.append('''
    <button class="carousel-btn prev" onclick="moveCarousel(-1)">‹</button>
    <button class="carousel-btn next" onclick="moveCarousel(1)">›</button>
    <div class="carousel-dots"></div>
    </div>''')  # carousel-container
    
    return "".join(html_parts)


def render_ask_vlm_result(content: dict, evidence_pages: list[int], base_path: Path) -> str:
    """Render ask_vlm result with the slide image and VLM answer."""
    deck = content.get("deck", "")
    slide = content.get("slide", "")
    question = content.get("question", "")
    answer = content.get("answer", "No answer")
    error = content.get("error")
    
    if error:
        return f'<div class="vlm-error">❌ {escape_html(error)}</div>'
    
    # Determine images base path
    project_root = None
    current = base_path
    while current.parent != current:
        if current.name == "COMSE6998-13":
            project_root = current
            break
        current = current.parent
    
    if project_root is None:
        project_root = Path("/home/pg2860/COMSE6998-13")
    
    images_base = project_root / "data" / "slidevqa" / "images"
    
    # Extract slide number
    slide_num = None
    if "_" in slide:
        parts = slide.split("_")
        if len(parts) >= 2:
            try:
                slide_num = int(parts[1])
            except ValueError:
                pass
    
    # Check if golden
    is_golden = slide_num is not None and slide_num in evidence_pages
    golden_class = " golden-slide" if is_golden else ""
    
    # Build image path
    img_path = images_base / deck / slide
    if img_path.exists():
        rel_path = f"../../../../../data/slidevqa/images/{deck}/{slide}"
    else:
        rel_path = ""
    
    html = f'''
    <div class="vlm-result{golden_class}">
        <div class="vlm-header">
            <span class="vlm-label">🔍 Vision Model Analysis</span>
            {f'<span class="golden-badge">🏆 Golden Evidence</span>' if is_golden else ''}
        </div>
        {f'<img src="{rel_path}" alt="Slide {slide_num}" class="vlm-image" loading="lazy" />' if rel_path else '<div class="vlm-image-placeholder">Image not found</div>'}
        <div class="vlm-qa">
            <div class="vlm-question"><strong>Question:</strong> {escape_html(question)}</div>
            <div class="vlm-answer"><strong>Answer:</strong> {escape_html(answer)}</div>
        </div>
        <div class="vlm-meta">
            <span>{escape_html(deck)}</span>
            <span>{escape_html(slide)}</span>
        </div>
    </div>
    '''
    
    return html


def generate_html_content(
    entries: list[dict],
    config: dict,
    question_data: dict,
    system_prompt: str,
    start_time: datetime | None,
    duration: float,
    log_path: Path,
) -> str:
    """Generate the full HTML content."""
    
    # Extract evidence pages for golden slide highlighting
    evidence_pages = question_data.get("evidence_pages", [])
    
    # Build timeline entries
    chat_html = []
    
    # Track current iteration/context
    last_role = None
    
    for entry in entries:
        level = entry.get("level", "")
        title = entry.get("title", "")
        content = entry.get("content")
        timestamp = entry.get("timestamp", "")
        
        # Format timestamp
        time_str = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = timestamp
        
        # Skip some system noise
        if level == "SYSTEM" or level == "DATA":
            continue
            
        # --- RENDER DIFFERENT MESSAGE TYPES ---
        
        # 1. User Message (Right aligned bubble)
        if title == "User Message" or (isinstance(content, dict) and content.get("type") == "HumanMessage"):
            msg_text = content if isinstance(content, str) else content.get("content", "")
            chat_html.append(f'''
            <div class="message user-message">
                <div class="message-bubble">{escape_html(msg_text)}</div>
                <div class="message-meta">User • {time_str}</div>
            </div>
            ''')
            last_role = "user"
            continue
            
        # 2. Agent Thought / Internal Monologue
        if level == "AGENT_THINK" and title == "Sending to LLM":
            # This is the prompt log, usually skip or show as collapsed thought
            continue
            
        if level == "AGENT_THINK" and title == "LLM Response":
            text = content.get("text", "")
            # Check if it's the final answer
            if "**FINAL ANSWER:" in text:
                answer = text.split("**FINAL ANSWER:", 1)[1].strip().strip("*")
                chat_html.append(f'''
                <div class="message bot-message final-answer">
                    <div class="avatar">🤖</div>
                    <div class="message-content">
                        <div class="final-answer-label">Final Answer</div>
                        <div class="final-answer-text">{escape_html(answer)}</div>
                    </div>
                </div>
                ''')
            elif text.strip():  # Only show non-empty thoughts
                chat_html.append(f'''
                <div class="message bot-message">
                    <div class="avatar">🤖</div>
                    <div class="message-content">
                        <div class="thought-bubble">
                            {escape_html(text)}
                        </div>
                    </div>
                </div>
                ''')
            # Skip empty responses (model just made a tool call with no text)
            last_role = "bot"
            continue
            
        # 3. Tool Calls (Styled blocks)
        if level == "TOOL_CALL":
            tool_name = title.replace("Calling tool: ", "")
            params = content
            
            # Format all parameters with retrieval-header style (same as results)
            params_html = ""
            if isinstance(params, dict) and params:
                param_items = []
                for key, value in params.items():
                    val_str = str(value)
                    # Replace technical terms with user-friendly ones
                    val_str = val_str.replace("nvembed", "neural")
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    param_items.append(f'<span class="query-param"><strong>{escape_html(key)}:</strong> {escape_html(val_str)}</span>')
                params_html = f'<div class="retrieval-header">{" | ".join(param_items)}</div>'
            
            chat_html.append(f'''
            <div class="message bot-message tool-action">
                <div class="avatar">🤖</div>
                <div class="message-content">
                    <div class="tool-call-card">
                        <div class="tool-call-header">
                            <span class="tool-icon">🛠️</span>
                            <span class="tool-name">Used <strong>{escape_html(tool_name)}</strong></span>
                            <span class="tool-status">Called</span>
                        </div>
                        {params_html}
                    </div>
                </div>
            </div>
            ''')
            continue
            
        # 4. Tool Results
        if level == "TOOL_RESULT":
            tool_name = title.split("(")[0].replace("Tool result: ", "").strip()
            
            # Render special visualizations for specific tools
            # All retrieval tools that return results with slides
            retrieval_tools = ["retrieve_and_rerank", "custom_query_search", "retrieve_nvembed", "retrieve_colpali_text", "retrieve_colpali_visual"]
            
            if tool_name == "list_tools" and isinstance(content, dict):
                tool_output_html = render_list_tools_result(content)
                open_details = ' open'  # Auto-expand list_tools
            elif tool_name in retrieval_tools and isinstance(content, dict):
                tool_output_html = render_retrieve_and_rerank_result(content, evidence_pages, log_path)
                open_details = ' open'  # Auto-expand retrieval results
            elif tool_name == "ask_vlm" and isinstance(content, dict):
                tool_output_html = render_ask_vlm_result(content, evidence_pages, log_path)
                open_details = ' open'  # Auto-expand ask_vlm
            else:
                tool_output_html = format_json_html(content)
                open_details = ''
            
            chat_html.append(f'''
            <div class="message bot-message tool-action">
                <div class="avatar">🤖</div>
                <div class="message-content">
                    <details class="tool-execution result"{open_details}>
                        <summary>
                            <span class="tool-icon">📥</span>
                            <span class="tool-name">Result from <strong>{escape_html(tool_name)}</strong></span>
                            <span class="tool-status">Received</span>
                        </summary>
                        <div class="tool-details">
                            {tool_output_html}
                        </div>
                    </details>
                </div>
            </div>
            ''')
            continue
            
        # 5. Skip internal system noise (Agent Plan, Decision routing)
        # These are internal logging, not useful for understanding the conversation
        if level in ["AGENT_PLAN", "DECISION"]:
            continue  # Hide these - they just clutter the view
        
        # 6. Generic Bot Message (MESSAGE level only, not internal events)
        if level == "MESSAGE" and title != "System Prompt" and title != "Message Trace":
            chat_html.append(f'''
            <div class="message system-event">
                <span class="system-icon">{get_entry_icon(level)}</span>
                <span class="system-text">{escape_html(title)}</span>
            </div>
            ''')
    
    # Build full HTML
    query_id = question_data.get("query_id", "?")
    question = question_data.get("question", "N/A")
    golden_answer = question_data.get("golden_answer", "N/A")
    evidence_pages = question_data.get("evidence_pages", [])
    model = config.get("model", "N/A")
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query {query_id} - Chat Trace</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-body: #f9fafb;
            --bg-chat: #ffffff;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --border-color: #e5e7eb;
            --accent-color: #2563eb;
            --user-bubble: #2563eb;
            --user-text: #ffffff;
            --bot-bubble-bg: #f3f4f6;
            --tool-bg: #ffffff;
            --tool-border: #e5e7eb;
            --success-color: #10b981;
            --font-main: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: var(--font-main);
            background: var(--bg-body);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        /* Header */
        .header {{
            background: var(--bg-chat);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            z-index: 10;
        }}
        
        .header-title {{
            font-weight: 600;
            font-size: 1.125rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .header-meta {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}
        
        .meta-item strong {{ color: var(--text-primary); font-weight: 500; }}
        
        /* Main Chat Area */
        .chat-container {{
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
        }}
        
        /* Question Block */
        .question-block {{
            background: var(--bg-chat);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        
        .question-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        
        .question-text {{
            font-size: 1.125rem;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 1rem;
        }}
        
        .golden-answer {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: #ecfdf5;
            color: #047857;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        
        /* Messages */
        .message {{
            display: flex;
            gap: 1rem;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }}
        
        .user-message {{
            align-self: flex-end;
            flex-direction: row-reverse;
        }}
        
        .bot-message {{
            align-self: flex-start;
        }}
        
        .avatar {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: var(--accent-color);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            flex-shrink: 0;
        }}
        
        .bot-message .avatar {{ background: var(--border-color); }}
        
        .message-content {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            min-width: 0; /* Fix flex overflow */
        }}
        
        .message-bubble {{
            background: var(--user-bubble);
            color: var(--user-text);
            padding: 0.75rem 1.25rem;
            border-radius: 18px 18px 4px 18px;
            font-size: 1rem;
            line-height: 1.5;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
        
        .thought-bubble {{
            background: var(--bg-chat);
            color: var(--text-secondary);
            padding: 0;
            font-size: 1rem;
            line-height: 1.6;
        }}
        
        .message-meta {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
            text-align: right;
        }}
        
        /* Tool Executions */
        .tool-action {{
            width: 100%;
            max-width: 100%;
        }}
        
        .tool-execution {{
            background: var(--bg-chat);
            border: 1px solid var(--tool-border);
            border-radius: 8px;
            overflow: hidden;
            width: 100%;
        }}
        
        .tool-call-card {{
            border: 1px solid var(--tool-border);
            border-radius: 8px;
            overflow: hidden;
            width: 100%;
        }}
        
        .tool-call-header {{
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--bot-bubble-bg);
        }}
        
        .tool-call-card .retrieval-header {{
            border-radius: 0;
            border-top: 1px solid var(--tool-border);
            border-left: none;
            border-right: none;
            border-bottom: none;
            margin: 0;
            background: white;
        }}
        
        .tool-execution summary {{
            padding: 0.75rem 1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--bot-bubble-bg);
            transition: background 0.2s;
            user-select: none;
        }}
        
        .tool-execution summary:hover {{
            background: #e5e7eb;
        }}
        
        .tool-execution[open] summary {{
            border-bottom: 1px solid var(--tool-border);
        }}
        
        .tool-name {{ font-weight: 500; flex: 1; font-family: var(--font-mono); font-size: 0.875rem; }}
        .tool-status {{ font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; }}
        
        .tool-params-inline {{
            margin-top: 0.5rem;
            padding: 0.5rem 0.75rem;
            background: #f5f5f5;
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }}
        
        .param-item strong {{
            color: var(--text-primary);
            font-weight: 500;
        }}
        
        .tool-details {{
            padding: 1rem;
            background: #ffffff;
        }}
        
        .json-block {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-family: var(--font-mono);
            font-size: 0.8125rem;
            white-space: pre-wrap;
        }}
        
        /* Final Answer */
        .final-answer {{
            width: 100%;
            max-width: 100%;
        }}
        
        .final-answer .message-content {{
            background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
            border: 1px solid var(--accent-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        
        .final-answer-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--accent-color);
            margin-bottom: 0.5rem;
        }}
        
        .final-answer-text {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        /* System Events */
        .system-event {{
            align-self: center;
            font-size: 0.8125rem;
            color: var(--text-secondary);
            background: var(--bot-bubble-bg);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            max-width: fit-content;
        }}
        
        /* Tools List Visualization */
        .tools-list {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .tool-item {{
            background: var(--bg-body);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .tool-item-header {{
            background: var(--bot-bubble-bg);
            padding: 0.75rem 1rem;
            font-weight: 600;
            font-family: var(--font-mono);
            font-size: 0.875rem;
            color: var(--accent-color);
        }}
        
        .tool-item-body {{
            padding: 1rem;
        }}
        
        .tool-item-desc {{
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }}
        
        .tool-item-meta {{
            font-size: 0.8125rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}
        
        /* Carousel */
        .carousel-container {{
            position: relative;
            width: 100%;
            max-width: 100%;
            background: var(--bg-body);
            border-radius: 12px;
            overflow: hidden;
            padding: 1rem;
        }}
        
        .carousel-slides {{
            display: flex;
            overflow-x: auto;
            scroll-behavior: smooth;
            scroll-snap-type: x mandatory;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }}
        
        .carousel-slides::-webkit-scrollbar {{
            display: none;
        }}
        
        .carousel-slide {{
            flex: 0 0 100%;
            scroll-snap-align: start;
            padding: 0.5rem;
            background: white;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            margin-right: 1rem;
        }}
        
        .carousel-slide.golden-slide {{
            border-color: #f59e0b;
            border-width: 3px;
            box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.2);
        }}
        
        .slide-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .slide-number {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
        }}
        
        .golden-badge {{
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .slide-score {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: var(--font-mono);
        }}
        
        .slide-image {{
            width: 100%;
            height: auto;
            max-height: 400px;
            object-fit: contain;
            border-radius: 6px;
            margin-bottom: 0.75rem;
            background: #fafafa;
        }}
        
        .slide-image-placeholder {{
            width: 100%;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--border-color);
            color: var(--text-secondary);
            border-radius: 6px;
            margin-bottom: 0.75rem;
        }}
        
        .slide-caption {{
            font-size: 0.875rem;
            line-height: 1.6;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
            border-left: 3px solid var(--accent-color);
            padding-left: 0.75rem;
        }}
        
        .slide-meta {{
            display: flex;
            gap: 1rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: var(--font-mono);
        }}
        
        .carousel-btn {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--border-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 1.5rem;
            color: var(--text-primary);
            transition: all 0.2s;
            z-index: 10;
        }}
        
        .carousel-btn:hover {{
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        
        .carousel-btn.prev {{
            left: 1rem;
        }}
        
        .carousel-btn.next {{
            right: 1rem;
        }}
        
        .carousel-dots {{
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1rem;
        }}
        
        /* VLM Result Display */
        .vlm-result {{
            background: white;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.5rem;
        }}
        
        .vlm-result.golden-slide {{
            border-color: #f59e0b;
            border-width: 3px;
            box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.2);
        }}
        
        .vlm-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .vlm-label {{
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--accent-color);
        }}
        
        .vlm-image {{
            width: 100%;
            height: auto;
            max-height: 400px;
            object-fit: contain;
            border-radius: 6px;
            margin-bottom: 1rem;
            background: #fafafa;
        }}
        
        .vlm-image-placeholder {{
            width: 100%;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--border-color);
            color: var(--text-secondary);
            border-radius: 6px;
            margin-bottom: 1rem;
        }}
        
        .vlm-qa {{
            margin-bottom: 0.75rem;
        }}
        
        .vlm-question {{
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            color: var(--text-primary);
        }}
        
        .vlm-answer {{
            font-size: 0.875rem;
            color: var(--text-primary);
            padding: 0.75rem;
            background: var(--bg-body);
            border-radius: 6px;
            border-left: 3px solid var(--accent-color);
        }}
        
        .vlm-meta {{
            display: flex;
            gap: 1rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: var(--font-mono);
        }}
        
        .vlm-error {{
            padding: 1rem;
            background: #fee;
            border: 1px solid #fcc;
            border-radius: 6px;
            color: #c33;
        }}
        
        /* Retrieval Query Info Header */
        .retrieval-header {{
            background: linear-gradient(135deg, var(--accent-color-light), #e8f4f8);
            border: 1px solid var(--accent-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            font-size: 0.875rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}
        
        .query-param {{
            color: var(--text-primary);
        }}
        
        .query-param strong {{
            color: var(--accent-color);
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <span>🔍 SlideVQA Trace</span>
            <span style="font-weight: normal; color: var(--text-secondary);">Query {query_id}</span>
        </div>
        <div class="header-meta">
            <div class="meta-item">Model: <strong>{escape_html(model)}</strong></div>
            <div class="meta-item">Duration: <strong>{duration:.2f}s</strong></div>
            <div class="meta-item">Tool Invocations: <strong>{sum(1 for e in entries if e.get('level') == 'TOOL_CALL')}</strong></div>
        </div>
    </div>
    
    <div class="chat-container">
        <!-- Question Section -->
        <div class="question-block">
            <div class="question-label">Current Question</div>
            <div class="question-text">{escape_html(question)}</div>
            <div class="golden-answer">
                <span>✅ Expected Answer:</span>
                <strong>{escape_html(golden_answer)}</strong>
            </div>
        </div>
        
        <!-- Chat History -->
        {"".join(chat_html)}
        
        <div style="height: 2rem;"></div> <!-- Spacing at bottom -->
    </div>
    
    <script>
        // Carousel navigation
        function moveCarousel(direction) {{
            const containers = document.querySelectorAll('.carousel-container');
            containers.forEach(container => {{
                const slides = container.querySelector('.carousel-slides');
                const slideWidth = slides.querySelector('.carousel-slide').offsetWidth;
                slides.scrollLeft += direction * (slideWidth + 16); // 16 = margin-right
            }});
        }}
    </script>
</body>
</html>'''
    
    return html


def generate_trace_for_query_dir(query_dir: Path | str) -> str:
    """Generate HTML trace for a query directory."""
    query_dir = Path(query_dir)
    log_path = query_dir / "agent_log.jsonl"
    
    if not log_path.exists():
        raise FileNotFoundError(f"No agent_log.jsonl found in {query_dir}")
    
    return generate_html_trace(log_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python html_trace_generator.py <path_to_agent_log.jsonl or query_dir>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_dir():
        output = generate_trace_for_query_dir(path)
    else:
        output = generate_html_trace(path)
    
    print(f"Generated Chat UI: {output}")
