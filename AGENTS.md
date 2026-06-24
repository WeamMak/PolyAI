# PolyAI Fursa - Agent Guidelines

This is an **educational project**. Students learn by reading, modifying, and extending real code.

## Project Overview

```
services/
  agent/    ← LangChain agent with manual tool-calling loop
  frontend/ ← Simple chat UI (talks to the agent)
  yolo/     ← YOLO object-detection microservice (FastAPI + Ultralytics)
```

---

## Course Content Reference

The course curriculum lives at: `github.com/alonitac/Fursa26`

When you are **unsure what concepts have been taught** - e.g., has the course covered `async`/`await`? TypeScript generics? React hooks? - use the GitHub search tools to look up relevant course materials in that repo before writing code.

Search for things like:
- Lesson or tutorial files mentioning the concept in question
- README files describing module or session goals
- Example code in the curriculum that sets the expected style and complexity level

If you find that a concept has not been taught yet, either avoid it or flag it clearly to the student.

---

## Skill Eval Reports

When using a skill from `.agents/skills/`, check whether that skill has an `evals/evals.json` file.

If it does:
- Select the eval cases relevant to the user's task.
- Use their assertions as a checklist while doing the work.
- Before finishing, write a dated report in that skill's `reports/` folder.
- The report must include the report date/time, the user prompt, which evals were used, which evals passed, which failed or were not run, and short evidence for each status.

If no evals exist for the skill, say that no skill evals were available.


## Architecture Constraints

### The LLM never sees image data

The LLM receives **text only**. Images are handled exclusively by the YOLO microservice.

- The `chat()` endpoint in `services/agent/app.py` must strip `image_base64` before building LangChain messages.
- The image is stored in `_current_image_b64` (a context variable) and passed directly to the `detect_objects` tool, which forwards it to the YOLO service.
- Do **not** add multimodal content (e.g. `image_url`) to `HumanMessage`. The model's role is conversation management, not vision.

---

## Coding Principles

### Keep it explicit, not magic
Prefer readable, step-by-step code over clever abstractions.
Students must be able to follow the execution flow line by line.

**Good:**
```python
response = llm_with_tools.invoke(messages)
for tool_call in response.tool_calls:
    result = TOOLS[tool_call["name"]].invoke(tool_call)
    messages.append(result)
```

**Avoid:**
```python
result = create_react_agent(llm, tools).invoke(state)
```

### Do not use high-level agent frameworks as a black box
`create_react_agent`, `AgentExecutor`, and similar wrappers hide the loop that students need to learn.
Implement the ReAct loop manually in `run_agent()` inside `services/agent/app.py`.
