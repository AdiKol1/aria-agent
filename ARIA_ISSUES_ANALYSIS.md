# Aria Voice Assistant - Comprehensive Issue Analysis

## Executive Summary

After exhaustive research, I've identified **11 critical issues** preventing Aria from being a reliable assistant. The core problem is that **Gemini-2.0-flash doesn't reliably call tools in voice mode**, and the fallback system designed to compensate has multiple bugs.

---

## Issue #1: GEMINI DOESN'T CALL TOOLS (ROOT CAUSE)

**Symptom**: User says "type hello", Gemini responds "Done" but nothing happens.

**Root Cause**: Gemini-2.0-flash-exp in Live API mode prioritizes natural speech over function calling. Even with properly registered tools, Gemini often chooses to SAY it's doing something rather than CALL the tool.

**Evidence**:
- `config.py:49-52`: "gemini-2.0-flash-exp has known function calling issues"
- `gemini_voice.py:142-143`: "tool_config is NOT supported by LiveConnectConfig"
- Logs show Gemini saying "I'll type that" without calling `type_text`

**Why this is critical**: This is the fundamental limitation causing all other issues.

---

## Issue #2: TWO DIFFERENT FALLBACK FUNCTIONS WITH DIFFERENT LOGIC

**Location**: `gemini_voice.py`

**The Problem**:
1. `_execute_fallback_action()` (lines 817-960) - IMMEDIATE fallback
   - Called by `_delayed_fallback()` - **DISABLED** via `DISABLE_IMMEDIATE_FALLBACK = True`
   - Limited patterns: apps, tabs, scroll, enter, mouse move
   - **Missing**: click, type, select all, copy, paste, undo, redo, delete, search

2. `_send_confabulation_correction()` (lines 1050-1420) - END-OF-TURN fallback
   - Called when confabulation detected - **ENABLED**
   - More comprehensive patterns
   - **But**: Many patterns missing normalization

**Why this is bad**: Code duplication, inconsistent behavior, hard to maintain.

---

## Issue #3: MISSING NORMALIZATION IN FALLBACK PATTERNS

**Location**: `gemini_voice.py` lines 1107-1330

**The Problem**: Many fallback patterns only check `request_lower` but NOT `request_normalized`. Fragmented speech like "ty pe" won't match "type".

**Patterns WITH normalization (correct)**:
- scroll (line 856)
- select all (line 1121)
- click (line 1201)
- move mouse (line 1234)

**Patterns MISSING normalization (BUGS)**:
| Line | Pattern | Should Also Check |
|------|---------|-------------------|
| 1107 | new tab | "newtab" in normalized |
| 1111 | close tab | "closetab" in normalized |
| 1115 | new window | "newwindow" in normalized |
| 1127 | copy | "copy" in normalized |
| 1131 | paste | "paste" in normalized |
| 1137 | undo | "undo" in normalized |
| 1141 | redo | "redo" in normalized |
| 1153 | open | "open" in normalized |
| 1162 | search | "search" in normalized |
| 1286 | type | "type" in normalized |
| 1318 | delete | "delete" in normalized |
| 1330 | send | "send" in normalized |

---

## Issue #4: `_last_action_request` RACE CONDITION

**Location**: `gemini_voice.py` lines 447-451, 697-713

**The Problem**:
1. User says "type hello" → `_last_action_request` set to "type hello"
2. Gemini confabulates → confabulation detected
3. User says "I don't see it" → `_last_user_request` updated
4. Fallback runs → uses `_last_action_request` (correct now after my fix)
5. BUT: `_last_action_request` might be stale from OLD request

**Current Fix Applied**: I added `_last_action_request` tracking, but there's still a potential race condition if the async task doesn't run immediately.

---

## Issue #5: ASYNC TASK RACE WITH STATE RESET

**Location**: `gemini_voice.py` lines 703, 712-713

```python
asyncio.create_task(self._send_confabulation_correction())  # Line 703
# ...
self._accumulated_text = ""  # Line 712
self._tool_called_this_turn = False  # Line 713 - RESETS BEFORE TASK COMPLETES!
```

**The Problem**: The confabulation correction is an async task, but the state is reset IMMEDIATELY after creating the task. If the task hasn't started yet, `_tool_called_this_turn` is already False.

**Impact**: The fallback might not execute properly because state is inconsistent.

---

## Issue #6: DUPLICATE PATTERN HANDLERS

**Location**: `gemini_voice.py`

**Duplicates Found**:
- "hit enter" handled at BOTH line 869 AND line 1326
- Scroll logic duplicated between `_execute_fallback_action` and `_send_confabulation_correction`
- App mappings duplicated in both functions

**Impact**: Inconsistent behavior, maintenance burden.

---

## Issue #7: PATTERN MATCHING FAILURES ON FRAGMENTED SPEECH

**Location**: `gemini_voice.py` lines 1205-1214

**The Problem**: Regex patterns can't match fragmented text.

**Example**:
- User says: "click in the text box"
- Transcribed as: "clic k in the text box"
- Pattern: `r'click\s+(?:on|in)\s+(?:the\s+)?(.+)'`
- This pattern looks for "click" but text has "clic k" - NO MATCH

**Current Fix Applied**: I added fallback extraction at lines 1217-1222, but it's hacky:
```python
target = request_lower.replace("clic", "").replace("k", "", 1).strip()
```

This only works for "clic k", not other fragmentations like "cl ick".

---

## Issue #8: CONFIG FLAG CHAOS

**Location**: `config.py` lines 57-64

```python
DISABLE_IMMEDIATE_FALLBACK = True        # Line 60
DISABLE_CONFABULATION_FALLBACK = False   # Line 61
DISABLE_FALLBACK_SYSTEM = True           # Line 64 (legacy)
```

**The Problem**: Three different disable flags, unclear which is active, legacy flag kept for "compatibility" but creates confusion.

---

## Issue #9: TRANSCRIPT BOUNDARY DETECTION IS FRAGILE

**Location**: `gemini_voice.py` lines 431-439

```python
if now - self._last_transcript_time > 5.0:
    # Reset for new request
```

**The Problem**: 5-second silence boundary is arbitrary. If user waits 4 seconds between requests, they're treated as ONE request, causing `_last_action_request` to contain merged text.

---

## Issue #10: TOOL DEFINITIONS WITHOUT HANDLERS

**Location**: `gemini_voice.py` lines 1712-2007 (definitions) vs `main.py` (handlers)

**The Problem**: ~50 tools defined but only ~15 have actual handlers.

**Tools WITHOUT handlers** (will return "Unknown tool" error):
- move_mouse_to_element
- click_element
- double_click
- right_click
- drag
- get_mouse_position
- wait
- find_element
- And many more...

---

## Issue #11: SYSTEM PROMPT NOT ENFORCED

**Location**: `main.py` lines 286-350

**The Problem**: System prompt tells Gemini to use tools, but Gemini ignores it in voice mode. The prompt says:
```
NEVER say "Done", "Opening", "Clicking" unless you ACTUALLY called a tool
```

But Gemini does exactly that because system prompts aren't enforced with same strength in Live API.

---

## THE CORRECT ARCHITECTURE

### Principle 1: FALLBACK MUST ALWAYS EXECUTE

When confabulation is detected, we MUST execute the action ourselves. Don't just send a correction message - ACTUALLY DO IT.

### Principle 2: UNIFIED FALLBACK FUNCTION

Merge `_execute_fallback_action()` and `_send_confabulation_correction()` into ONE comprehensive function with ALL patterns.

### Principle 3: CONSISTENT NORMALIZATION

EVERY pattern must check BOTH `request_lower` AND `request_normalized`:
```python
if "keyword" in request_lower or "keyword" in request_normalized:
```

### Principle 4: ROBUST FRAGMENTATION HANDLING

Instead of hacky string replacement, use a proper approach:
```python
def normalize_speech(text: str) -> str:
    """Remove all spaces to handle fragmented speech."""
    return text.lower().replace(' ', '')
```

Then match against normalized keywords.

### Principle 5: CLEAR STATE MANAGEMENT

Don't reset state until AFTER fallback completes:
```python
# Wait for fallback to complete before resetting
await self._send_confabulation_correction()
self._accumulated_text = ""
self._tool_called_this_turn = False
```

### Principle 6: SINGLE SOURCE OF CONFIG

One flag for fallback:
```python
FALLBACK_ENABLED = True  # That's it. One flag.
```

### Principle 7: COMPLETE TOOL HANDLERS

Every defined tool must have a handler. Remove tools that aren't implemented.

---

## IMPLEMENTATION PLAN

1. **Create unified fallback function** with ALL patterns and proper normalization
2. **Fix async race condition** by awaiting the fallback task
3. **Add normalization to ALL patterns**
4. **Remove duplicate code** between the two fallback functions
5. **Simplify config** to single flag
6. **Complete missing tool handlers** or remove undefined tools
7. **Test with fragmented speech** scenarios

---

## FILES TO MODIFY

| File | Changes |
|------|---------|
| `gemini_voice.py` | Unified fallback, fix race condition, add normalization |
| `config.py` | Simplify to single fallback flag |
| `main.py` | Complete tool handlers |

---

## SUCCESS CRITERIA

1. When user says "type hello", text IS typed (not just announced)
2. Fragmented speech "ty pe hel lo" works correctly
3. No duplicate action execution
4. Clear logging of what fallback did
5. Response time under 2 seconds for simple commands
6. ZERO confabulations - never claim an action without doing it
