"""Intent Parser - Core parsing logic for natural language commands.

This module handles the critical task of converting natural language user input
into structured Intent objects. It includes:
- Speech normalization (handling fragmented speech from voice recognition)
- Pattern matching for intent classification
- Target extraction from command text
- Confidence scoring

The parser is designed to be robust against common speech recognition errors
and variations in how users express their commands.
"""
import re
from typing import Optional, Dict, Any, List, Tuple
from ..intents.base import Intent, IntentType


# =============================================================================
# SPEECH NORMALIZATION
# =============================================================================

# Common words that may be fragmented by speech recognition
FRAGMENTED_WORDS = {
    # Action words
    "clic k": "click",
    "cli ck": "click",
    "cl ick": "click",
    "scro ll": "scroll",
    "scr oll": "scroll",
    "ty pe": "type",
    "typ e": "type",
    "sel ect": "select",
    "se lect": "select",
    "doub le": "double",
    "dou ble": "double",
    "op en": "open",
    "o pen": "open",
    "clo se": "close",
    "cl ose": "close",
    "nav igate": "navigate",
    "navi gate": "navigate",
    "laun ch": "launch",
    "lau nch": "launch",
    "rem ember": "remember",
    "remem ber": "remember",
    "rec all": "recall",
    "re call": "recall",
    "for get": "forget",
    "forg et": "forget",
    # Common targets
    "safa ri": "safari",
    "chro me": "chrome",
    "fire fox": "firefox",
    "ter minal": "terminal",
    "fin der": "finder",
    "spo tify": "spotify",
    "sla ck": "slack",
    # Common words
    "hel lo": "hello",
    "he llo": "hello",
    "hell o": "hello",
    "do wn": "down",
    "dow n": "down",
    "u p": "up",
    "le ft": "left",
    "rig ht": "right",
    "bac k": "back",
    "ba ck": "back",
    "for ward": "forward",
    "ent er": "enter",
    "en ter": "enter",
    "esc ape": "escape",
    "es cape": "escape",
    "del ete": "delete",
    "de lete": "delete",
    "spa ce": "space",
    "co py": "copy",
    "cop y": "copy",
    "pas te": "paste",
    "pa ste": "paste",
    "un do": "undo",
    "und o": "undo",
    "re do": "redo",
    "red o": "redo",
    "sa ve": "save",
    "sav e": "save",
    "qui t": "quit",
    "qu it": "quit",
    "exi t": "exit",
    "ex it": "exit",
    "new tab": "new tab",
    "clo se tab": "close tab",
    "clos e tab": "close tab",
    # More common words from live testing
    "no w": "now",
    "n ow": "now",
    "star t": "start",
    "sta rt": "start",
    "st art": "start",
    "ty ping": "typing",
    "typ ing": "typing",
    "typin g": "typing",
    "wri te": "write",
    "writ e": "write",
    "wr ite": "write",
    "plea se": "please",
    "ple ase": "please",
    "gre at": "great",
    "grea t": "great",
    "tha nk": "thank",
    "than k": "thank",
    "yo u": "you",
    "y ou": "you",
    "wha t": "what",
    "wh at": "what",
    "not es": "notes",
    "no tes": "notes",
    "n otes": "notes",
    "not e": "note",
    "no te": "note",
    # More common fragments
    "ne w": "new",
    "n ew": "new",
    "te xt": "text",
    "tex t": "text",
    "cur rently": "currently",
    "current ly": "currently",
    "th is": "this",
    "thi s": "this",
    "tha t": "that",
    "th at": "that",
    "ju st": "just",
    "jus t": "just",
    "clo se": "close",
    "clos e": "close",
    "ope n": "open",
    "o pen": "open",
    "del ete": "delete",
    "dele te": "delete",
    "se lect": "select",
    "selec t": "select",
    "al l": "all",
    # More fragments from live testing
    "ye s": "yes",
    "y es": "yes",
    "nee d": "need",
    "ne ed": "need",
    "but ton": "button",
    "butt on": "button",
    "bu tton": "button",
    # Box (from voice testing - "text box")
    "bo x": "box",
    "b ox": "box",
    "clou d": "cloud",
    "clo ud": "cloud",
    "cl oud": "cloud",
    # Icon (from voice testing)
    "ico n": "icon",
    "ic on": "icon",
    "i con": "icon",
    # Dock (from voice testing)
    "doc k": "dock",
    "do ck": "dock",
    "d ock": "dock",
    "scree n": "screen",
    "scre en": "screen",
    "win dow": "window",
    "windo w": "window",
    "brow ser": "browser",
    "browse r": "browser",
    "sear ch": "search",
    "searc h": "search",
    "se arch": "search",
    # Mouse and movement
    "mo use": "mouse",
    "mou se": "mouse",
    "mo uth": "mouse",  # common speech recognition error
    "mous e": "mouse",
    "mo ve": "move",
    "mov e": "move",
    "fir st": "first",
    "firs t": "first",
    "fi rst": "first",
    "doub le": "double",
    "dou ble": "double",
    "doubl e": "double",
    "rig ht": "right",
    "righ t": "right",
    "dra g": "drag",
    "dr ag": "drag",
    "dro p": "drop",
    "dr op": "drop",
    # URL/domain fragments (from voice testing)
    "compa ss": "compass",
    "comp ass": "compass",
    "com pass": "compass",
    "goo gle": "google",
    "goog le": "google",
    "go ogle": "google",
    "face book": "facebook",
    "faceb ook": "facebook",
    "you tube": "youtube",
    "yout ube": "youtube",
    "git hub": "github",
    "gith ub": "github",
    "twit ter": "twitter",
    "twitt er": "twitter",
    "ama zon": "amazon",
    "amaz on": "amazon",
    "net flix": "netflix",
    "netfl ix": "netflix",
    "lin ked": "linked",
    "link ed": "linked",
    ". com": ".com",
    ". org": ".org",
    ". net": ".net",
    ". io": ".io",
    # More from voice testing
    "que stion": "question",
    "ques tion": "question",
    "Cla ude": "Claude",
    "Clau de": "Claude",
    "cl aude": "claude",
    "cla ude": "claude",
    "Ar ya": "Arya",
    "Ary a": "Arya",
    "Ar ia": "Aria",
    "Ari a": "Aria",
    "vir tual": "virtual",
    "virtu al": "virtual",
    "ass istant": "assistant",
    "assis tant": "assistant",
    "as sistant": "assistant",
    # More common words
    "ex plain": "explain",
    "expl ain": "explain",
    "be cause": "because",
    "becau se": "because",
    "im prove": "improve",
    "impro ve": "improve",
    "ef fective": "effective",
    "effec tive": "effective",
    "con trol": "control",
    "cont rol": "control",
    "comp uter": "computer",
    "compu ter": "computer",
    # More from testing
    "ha te": "hate",
    "hat e": "hate",
    "fo llow": "follow",
    "fol low": "follow",
    "did n't": "didn't",
    "does n't": "doesn't",
    "can n't": "can't",
    "won n't": "won't",
    "ver ify": "verify",
    "veri fy": "verify",
    "actua lly": "actually",
    "actual ly": "actually",
    "supp osed": "supposed",
    "suppo sed": "supposed",
}

# Common speech-to-text errors and their corrections
SPEECH_CORRECTIONS = {
    # Click variations
    "clique": "click",
    "clik": "click",
    "klick": "click",
    "clck": "click",
    # Scroll variations
    "scrawl": "scroll",
    "scrolll": "scroll",
    # Type variations
    "tipe": "type",
    "tyep": "type",
    # Open variations
    "oppen": "open",
    "opn": "open",
    # Close variations
    "cloes": "close",
    "clsoe": "close",
    # Other common errors
    "launche": "launch",
    "naviagate": "navigate",
    "naviaget": "navigate",
    "remmeber": "remember",
    "remeber": "remember",
    "recal": "recall",
    # Dock variations (common speech errors)
    "duck": "dock",
    "duk": "dock",
    "duc": "dock",
    # Icon variations
    "icone": "icon",
    "ikon": "icon",
}


def clean_voice_input(text: str) -> str:
    """Clean voice input by removing conversational prefixes/suffixes.

    Voice transcripts often contain conversational context that interferes
    with action parsing. For example:
    - "Yes, can you open notes?" -> "open notes"
    - "Open the new tab. Go" -> "open the new tab"
    - "Can you please scroll down?" -> "scroll down"

    Args:
        text: The raw transcript text (possibly accumulated).

    Returns:
        Cleaned text with just the action command.
    """
    if not text:
        return ""

    text = text.strip()

    # First, normalize fragmented speech (e.g., "no w" -> "now", "star t" -> "start")
    text = normalize_speech(text)

    # Common conversational prefixes to remove (order matters - more specific first)
    prefixes_to_remove = [
        # Addressing (most specific first)
        r'^hey\s+aria\s*[,.]?\s*',
        r'^okay\s+aria\s*[,.]?\s*',
        r'^hi\s+aria\s*[,.]?\s*',
        r'^hello\s+aria\s*[,.]?\s*',
        r'^aria\s*[,.]?\s*',
        r'^hey\s*[,.]?\s*',
        r'^hi\s*[,.]?\s*',
        r'^hello\s*[,.]?\s*',
        # Affirmations
        r'^(?:yes|yeah|yep|sure|alright|right|great|perfect|good|nice|awesome|cool)\s*[,.]?\s*',
        r'^okay\s*[,.]?\s*',
        r'^ok\s*[,.]?\s*',
        # Repetition/clarification (when user repeats their command)
        r'^(?:i said|i asked you to|i told you to|i wanted you to)\s+',
        # Questions/requests
        r'^(?:can you|could you|would you|will you)\s+(?:please\s+)?',
        r'^please\s+(?:can you|could you|would you|will you)\s+',
        r'^(?:i want you to|i need you to|i\'d like you to)\s+',
        r'^(?:go ahead and|just|now)\s+',
        # Politeness
        r'^please\s+',
        r'\s+please$',  # trailing please
    ]

    # Apply prefix removal (multiple passes)
    for _ in range(3):
        original = text
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
        if text == original:
            break

    # Handle URLs specially - don't split on dots within URLs
    # Check if text contains a URL pattern
    url_pattern = r'(?:https?://)?(?:www\.)?[\w.-]+\.\w{2,}'
    has_url = re.search(url_pattern, text, re.IGNORECASE)

    if has_url:
        # Split more carefully - only on sentence-ending punctuation followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    else:
        # Normal sentence splitting
        sentences = re.split(r'[.!?]+', text)

    action_keywords = ['open', 'click', 'scroll', 'type', 'close', 'go to', 'navigate',
                       'copy', 'paste', 'undo', 'redo', 'save', 'new tab', 'press',
                       'launch', 'start', 'quit', 'exit', 'search', 'find', 'move',
                       'double click', 'right click', 'drag', 'drop', 'select']

    # Clean sentence prefixes (affirmations, negations, etc.)
    # Use word boundary \b to prevent "no" from matching "now"
    sentence_prefixes = [
        r'^(?:yes|yeah|yep|no|nope|okay|ok|sure|well|so|and|but|then|great|perfect|good|nice|awesome|cool|alright|right)\b\s*[,.]?\s*',
        r'^now\s*[,.]?\s+',  # "now" must be followed by space after optional comma/period
        r'^(?:i think|i believe|i want|i need)\s+',
        r'^(?:i said|i asked you to|i told you to)\s+',  # When user repeats their command
    ]

    # Patterns that contain an action - extract just the action part
    # e.g., "you need to open X" -> "open X"
    action_extraction_patterns = [
        r'(?:you\s+)?(?:need|want|have)\s+to\s+',  # "you need to open" -> "open"
        r'(?:can|could|would|will)\s+you\s+(?:please\s+)?',  # "can you open" -> "open"
        r'(?:please\s+)?(?:go\s+ahead\s+and\s+)',  # "go ahead and open" -> "open"
        r'(?:try|use)\s+(?:the\s+)?',  # "use the copy button" -> "copy button"
        r'(?:i\s+said|i\s+asked\s+you\s+to|i\s+told\s+you\s+to)\s+',  # "I said open" -> "open"
    ]

    def clean_sentence(s: str) -> str:
        """Remove conversational prefixes from a sentence."""
        s = s.strip()
        for pattern in sentence_prefixes:
            s = re.sub(pattern, '', s, flags=re.IGNORECASE).strip()
        return s

    def extract_action(s: str) -> str:
        """Extract action command from sentence patterns."""
        s = s.strip()
        for pattern in action_extraction_patterns:
            match = re.search(pattern, s, flags=re.IGNORECASE)
            if match:
                # Return everything after the matched pattern
                extracted = s[match.end():].strip()
                if extracted:
                    return extracted
        return s

    # First pass: look for sentences that START with an action keyword (most reliable)
    for sentence in sentences:
        sentence = clean_sentence(sentence)
        if not sentence:
            continue
        # Try to extract action from patterns like "you need to open X"
        sentence = extract_action(sentence)
        sentence_lower = sentence.lower()
        for keyword in action_keywords:
            if sentence_lower.startswith(keyword + ' ') or sentence_lower == keyword:
                # Found a sentence that starts with an action keyword
                if not has_url:
                    words = sentence.split()
                    if len(words) > 2 and len(words[-1]) <= 3 and words[-1].lower() in ['go', 'the', 'a', 'i', 'it', 'is']:
                        sentence = ' '.join(words[:-1])
                return sentence.strip()

    # Second pass: look for sentences that CONTAIN an action keyword (less reliable)
    for sentence in sentences:
        sentence = clean_sentence(sentence)
        if not sentence:
            continue
        # Try to extract action from patterns
        sentence = extract_action(sentence)
        sentence_lower = sentence.lower()
        for keyword in action_keywords:
            if keyword in sentence_lower:
                # Found an action sentence - clean it up
                # Remove trailing fragments like single words (but not if it's part of a URL)
                if not has_url:
                    words = sentence.split()
                    # If last word is very short and standalone, it might be a fragment
                    if len(words) > 2 and len(words[-1]) <= 3 and words[-1].lower() in ['go', 'the', 'a', 'i', 'it', 'is']:
                        sentence = ' '.join(words[:-1])
                return sentence.strip()

    # No clear action found - return first non-empty sentence
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            return sentence

    return text.strip()


def normalize_speech(text: str) -> str:
    """Normalize speech input by fixing fragmented words and common errors.

    Speech recognition often produces fragmented words where spaces are
    incorrectly inserted (e.g., "clic k" instead of "click"). This function
    reassembles these fragments and corrects common speech-to-text errors.

    Args:
        text: The raw speech recognition output.

    Returns:
        Normalized text with corrections applied.

    Example:
        >>> normalize_speech("clic k on chro me")
        'click on chrome'
        >>> normalize_speech("clique the button")
        'click the button'
    """
    if not text:
        return ""

    # Convert to lowercase for consistent processing
    normalized = text.lower().strip()

    # First, normalize multiple spaces to single spaces (so patterns match)
    normalized = re.sub(r'\s+', ' ', normalized)

    # Fix fragmented words (multi-pass to catch nested fragments)
    for _ in range(3):  # Multiple passes for complex fragmentation
        for fragment, correction in FRAGMENTED_WORDS.items():
            normalized = normalized.replace(fragment, correction)

    # Fix common speech recognition errors
    words = normalized.split()
    corrected_words = []
    for word in words:
        corrected_words.append(SPEECH_CORRECTIONS.get(word, word))
    normalized = " ".join(corrected_words)

    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

# Patterns for each intent type, ordered by specificity
INTENT_PATTERNS: Dict[IntentType, List[Tuple[str, float]]] = {
    IntentType.CLICK: [
        (r'^click\s+on\s+(?:the\s+)?(.+)$', 1.0),
        (r'^click\s+(?:the\s+)?(.+)$', 0.95),
        (r'^tap\s+(?:on\s+)?(?:the\s+)?(.+)$', 0.9),
        (r'^press\s+(?:on\s+)?(?:the\s+)?(.+)$', 0.85),
        (r'^select\s+(?:the\s+)?(.+)$', 0.8),
        (r'^hit\s+(?:the\s+)?(.+)$', 0.75),
        (r'^double[\s-]?click\s+(?:on\s+)?(?:the\s+)?(.+)$', 0.95),
        (r'^right[\s-]?click\s+(?:on\s+)?(?:the\s+)?(.+)$', 0.95),
    ],
    IntentType.OPEN: [
        (r'^open\s+(?:the\s+)?(?:app\s+)?(.+)$', 1.0),
        (r'^launch\s+(?:the\s+)?(?:app\s+)?(.+)$', 0.95),
        (r'^start\s+(?:the\s+)?(?:app\s+)?(.+)$', 0.9),
        (r'^run\s+(?:the\s+)?(?:app\s+)?(.+)$', 0.85),
        (r'^open\s+up\s+(.+)$', 0.9),
        (r'^fire\s+up\s+(.+)$', 0.8),
        (r'^bring\s+up\s+(.+)$', 0.75),
    ],
    IntentType.TYPE: [
        (r'^type\s+["\']?(.+?)["\']?$', 1.0),
        (r'^write\s+["\']?(.+?)["\']?$', 0.95),
        (r'^enter\s+["\']?(.+?)["\']?$', 0.9),
        (r'^input\s+["\']?(.+?)["\']?$', 0.85),
        (r'^type\s+(?:in\s+)?["\']?(.+?)["\']?$', 0.95),
        (r'^write\s+out\s+["\']?(.+?)["\']?$', 0.9),
    ],
    IntentType.SCROLL: [
        (r'^scroll\s+up(?:\s+(.+))?$', 1.0),
        (r'^scroll\s+down(?:\s+(.+))?$', 1.0),
        (r'^page\s+up$', 0.95),
        (r'^page\s+down$', 0.95),
        (r'^scroll\s+to\s+(?:the\s+)?(top|bottom)$', 0.9),
        (r'^scroll\s+left(?:\s+(.+))?$', 0.9),
        (r'^scroll\s+right(?:\s+(.+))?$', 0.9),
        (r'^go\s+up(?:\s+a\s+(?:bit|page))?$', 0.7),
        (r'^go\s+down(?:\s+a\s+(?:bit|page))?$', 0.7),
    ],
    IntentType.CLOSE: [
        (r'^close\s+(?:the\s+)?(.+)$', 1.0),
        (r'^quit\s+(?:the\s+)?(.+)$', 0.95),
        (r'^exit\s+(?:the\s+)?(.+)$', 0.9),
        (r'^kill\s+(?:the\s+)?(.+)$', 0.85),
        (r'^terminate\s+(?:the\s+)?(.+)$', 0.8),
        (r'^shut\s+down\s+(?:the\s+)?(.+)$', 0.8),
        (r'^close$', 0.7),  # Close current window
    ],
    IntentType.TAB: [
        (r'^new\s+tab$', 1.0),
        (r'^open\s+(?:a\s+)?new\s+tab$', 0.95),
        (r'^close\s+(?:this\s+)?tab$', 1.0),
        (r'^close\s+(?:the\s+)?current\s+tab$', 0.95),
        (r'^switch\s+(?:to\s+)?(?:the\s+)?next\s+tab$', 0.95),
        (r'^next\s+tab$', 0.9),
        (r'^switch\s+(?:to\s+)?(?:the\s+)?previous\s+tab$', 0.95),
        (r'^previous\s+tab$', 0.9),
        (r'^prev\s+tab$', 0.85),
        (r'^switch\s+(?:to\s+)?tab\s+(\d+)$', 0.95),
        (r'^go\s+to\s+tab\s+(\d+)$', 0.9),
        (r'^(?:re)?open\s+(?:the\s+)?last\s+(?:closed\s+)?tab$', 0.9),
    ],
    IntentType.KEYBOARD: [
        # Single key shortcuts
        (r'^copy$', 1.0),
        (r'^paste$', 1.0),
        (r'^cut$', 1.0),
        (r'^undo$', 1.0),
        (r'^redo$', 1.0),
        (r'^save$', 1.0),
        (r'^delete$', 0.95),
        (r'^select\s+all$', 1.0),
        # Explicit shortcuts
        (r'^(?:do\s+)?(?:command|cmd|ctrl|control)[\s+\-](.+)$', 0.95),
        (r'^press\s+(?:command|cmd|ctrl|control)[\s+\-](.+)$', 0.95),
        (r'^(?:hit\s+)?(?:command|cmd|ctrl|control)[\s+\-](.+)$', 0.9),
        # Key presses
        (r'^press\s+(enter|return|escape|esc|tab|space|backspace|delete)$', 0.95),
        (r'^hit\s+(enter|return|escape|esc|tab|space|backspace|delete)$', 0.9),
        # Find
        (r'^find$', 0.9),
        (r'^search$', 0.85),
    ],
    IntentType.NAVIGATE: [
        (r'^go\s+to\s+(.+)$', 1.0),
        (r'^navigate\s+to\s+(.+)$', 0.95),
        (r'^visit\s+(.+)$', 0.9),
        (r'^open\s+(https?://\S+)$', 0.95),
        (r'^open\s+(\S+\.\S+)$', 0.85),  # Matches domain-like strings
        (r'^browse\s+(?:to\s+)?(.+)$', 0.8),
        (r'^load\s+(.+)$', 0.75),
    ],
    IntentType.MEMORY: [
        (r'^remember\s+(?:that\s+)?(.+)$', 1.0),
        (r'^don\'?t\s+forget\s+(?:that\s+)?(.+)$', 0.95),
        (r'^keep\s+in\s+mind\s+(?:that\s+)?(.+)$', 0.9),
        (r'^note\s+(?:that\s+)?(.+)$', 0.85),
        (r'^save\s+(?:the\s+)?fact\s+(?:that\s+)?(.+)$', 0.9),
        (r'^recall\s+(.+)$', 1.0),
        (r'^what\s+do\s+(?:you\s+)?(?:i\s+)?know\s+about\s+(.+)$', 0.95),
        (r'^do\s+you\s+remember\s+(.+)$', 0.9),
        (r'^forget\s+(?:about\s+)?(.+)$', 0.95),
        (r'^list\s+(?:all\s+)?memories$', 0.95),
        (r'^show\s+(?:all\s+)?memories$', 0.9),
        (r'^what\s+do\s+you\s+know\s+about\s+me$', 0.95),
    ],
}

# Keywords that trigger intent types (for fuzzy matching)
INTENT_KEYWORDS: Dict[IntentType, List[str]] = {
    IntentType.CLICK: ["click", "tap", "press", "select", "hit"],
    IntentType.OPEN: ["open", "launch", "start", "run", "fire up", "bring up"],
    IntentType.TYPE: ["type", "write", "enter", "input"],
    IntentType.SCROLL: ["scroll", "page up", "page down"],
    IntentType.CLOSE: ["close", "quit", "exit", "kill", "terminate"],
    IntentType.TAB: ["tab", "new tab", "close tab", "next tab", "previous tab"],
    IntentType.KEYBOARD: ["copy", "paste", "undo", "redo", "cut", "save", "delete", "select all", "cmd", "command", "ctrl"],
    IntentType.NAVIGATE: ["go to", "navigate", "visit", "browse"],
    IntentType.MEMORY: ["remember", "recall", "forget", "memory", "memories", "know about"],
}


# =============================================================================
# TARGET EXTRACTION
# =============================================================================

def extract_target(text: str, action: IntentType) -> Optional[str]:
    """Extract the target from command text based on the action type.

    This function identifies and extracts the object/target of an action
    from the user's command. For example, extracting "Chrome" from
    "click on the Chrome icon in the dock".

    Args:
        text: The normalized command text.
        action: The identified intent type.

    Returns:
        The extracted target string, or None if no target found.

    Example:
        >>> extract_target("click on the Chrome icon", IntentType.CLICK)
        'Chrome'
        >>> extract_target("open finder", IntentType.OPEN)
        'finder'
    """
    if not text or action == IntentType.UNKNOWN:
        return None

    text = text.lower().strip()

    # Try to match against patterns for this intent type
    if action in INTENT_PATTERNS:
        for pattern, _ in INTENT_PATTERNS[action]:
            match = re.match(pattern, text, re.IGNORECASE)
            if match and match.groups():
                target = match.group(1)
                if target:
                    # Clean up the target
                    target = _clean_target(target, action)
                    return target

    # Fallback: extract based on action keywords
    target = _extract_target_fallback(text, action)
    return target


def _clean_target(target: str, action: IntentType) -> str:
    """Clean and normalize an extracted target string.

    Args:
        target: The raw extracted target.
        action: The intent type for context-specific cleaning.

    Returns:
        Cleaned target string.
    """
    if not target:
        return ""

    target = target.strip()

    # Special handling for URLs (NAVIGATE action)
    # Extract just the URL/domain from noisy transcript text
    if action == IntentType.NAVIGATE:
        # First, normalize fragmented URLs (e.g., "compa ss.com" -> "compass.com")
        target = normalize_speech(target)

        # Try to extract a clean URL or domain from the text
        # Pattern: domain.tld or full URL
        url_pattern = r'((?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?)'
        url_match = re.search(url_pattern, target, re.IGNORECASE)
        if url_match:
            target = url_match.group(1)
            # Clean up any trailing punctuation or noise
            target = re.sub(r'[.,;:!?\s]+$', '', target)
            # Remove trailing words that got attached (e.g., "compass.com. no" -> "compass.com")
            target = re.sub(r'\.\s+\w+$', '', target)
            return target

        # If no URL pattern found, clean up common noise
        # Remove trailing sentence fragments
        target = re.split(r'\.\s+', target)[0]  # Take first sentence
        target = re.sub(r'[.,;:!?\s]+$', '', target)  # Clean trailing punctuation

    # Remove common noise words at the end
    noise_suffixes = [
        " icon", " button", " app", " application", " window",
        " in the dock", " from the dock", " on the dock",
        " in the menu", " from the menu", " on the menu",
        " please", " for me",
    ]

    for suffix in noise_suffixes:
        if target.lower().endswith(suffix):
            target = target[:-len(suffix)].strip()

    # Remove leading noise words
    noise_prefixes = ["the ", "a ", "an "]
    for prefix in noise_prefixes:
        if target.lower().startswith(prefix):
            target = target[len(prefix):].strip()

    return target


def _extract_target_fallback(text: str, action: IntentType) -> Optional[str]:
    """Fallback target extraction when pattern matching fails.

    Args:
        text: The command text.
        action: The intent type.

    Returns:
        Extracted target or None.
    """
    # Remove action keywords to isolate target
    keywords = INTENT_KEYWORDS.get(action, [])

    remaining = text
    for keyword in keywords:
        # Remove keyword and common prepositions
        patterns = [
            f"^{keyword}\\s+(?:on\\s+)?(?:the\\s+)?",
            f"^{keyword}\\s+",
        ]
        for pattern in patterns:
            remaining = re.sub(pattern, "", remaining, flags=re.IGNORECASE)

    remaining = remaining.strip()
    if remaining and remaining != text:
        return _clean_target(remaining, action)

    return None


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def calculate_confidence(text: str, action: IntentType) -> float:
    """Calculate confidence score for an intent match.

    This function evaluates how well the input text matches the identified
    intent type, returning a score from 0.0 to 1.0.

    Args:
        text: The normalized command text.
        action: The identified intent type.

    Returns:
        Confidence score from 0.0 (no match) to 1.0 (exact match).

    Example:
        >>> calculate_confidence("click on Chrome", IntentType.CLICK)
        1.0
        >>> calculate_confidence("tap the button", IntentType.CLICK)
        0.9
    """
    if not text or action == IntentType.UNKNOWN:
        return 0.0

    text = text.lower().strip()
    max_confidence = 0.0

    # Check pattern matches
    if action in INTENT_PATTERNS:
        for pattern, base_confidence in INTENT_PATTERNS[action]:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                max_confidence = max(max_confidence, base_confidence)

    # Check keyword presence as fallback
    if max_confidence == 0.0 and action in INTENT_KEYWORDS:
        keywords = INTENT_KEYWORDS[action]
        for keyword in keywords:
            if keyword in text:
                # Partial confidence for keyword match
                max_confidence = max(max_confidence, 0.5)
                break

    return max_confidence


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

def _classify_intent(text: str) -> Tuple[IntentType, float]:
    """Classify the intent type of a command.

    Args:
        text: The normalized command text.

    Returns:
        Tuple of (IntentType, confidence score).
    """
    if not text:
        return IntentType.UNKNOWN, 0.0

    text = text.lower().strip()
    best_match = IntentType.UNKNOWN
    best_confidence = 0.0

    # Check each intent type's patterns
    for intent_type, patterns in INTENT_PATTERNS.items():
        for pattern, base_confidence in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                # Adjust confidence based on match quality
                confidence = base_confidence

                # Boost confidence for exact matches
                if match.group(0) == text:
                    confidence = min(1.0, confidence + 0.05)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = intent_type

    # If no pattern match, try keyword matching
    if best_confidence == 0.0:
        for intent_type, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if text.startswith(keyword) or f" {keyword}" in text:
                    # Lower confidence for keyword-only matches
                    confidence = 0.5
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = intent_type
                    break

    return best_match, best_confidence


# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================

def _extract_params(text: str, action: IntentType) -> Dict[str, Any]:
    """Extract additional parameters from command text.

    Args:
        text: The normalized command text.
        action: The identified intent type.

    Returns:
        Dictionary of extracted parameters.
    """
    params: Dict[str, Any] = {}
    text_lower = text.lower()

    if action == IntentType.CLICK:
        # Check for click modifiers
        if "double" in text_lower:
            params["double_click"] = True
        if "right" in text_lower:
            params["right_click"] = True

    elif action == IntentType.SCROLL:
        # Extract scroll direction and amount
        if "up" in text_lower:
            params["direction"] = "up"
        elif "down" in text_lower:
            params["direction"] = "down"
        elif "left" in text_lower:
            params["direction"] = "left"
        elif "right" in text_lower:
            params["direction"] = "right"
        elif "top" in text_lower:
            params["direction"] = "top"
        elif "bottom" in text_lower:
            params["direction"] = "bottom"

        # Extract amount if specified
        amount_match = re.search(r'(\d+)\s*(?:pixels?|px|lines?|pages?)?', text_lower)
        if amount_match:
            params["amount"] = int(amount_match.group(1))
        elif "a bit" in text_lower or "little" in text_lower:
            params["amount"] = 100
        elif "a lot" in text_lower or "much" in text_lower:
            params["amount"] = 500
        elif "page" in text_lower:
            params["amount"] = 300

    elif action == IntentType.TAB:
        # Extract tab number if specified
        tab_match = re.search(r'tab\s+(\d+)', text_lower)
        if tab_match:
            params["tab_number"] = int(tab_match.group(1))

        # Determine tab action
        if "new" in text_lower:
            params["action"] = "new"
        elif "close" in text_lower:
            params["action"] = "close"
        elif "next" in text_lower:
            params["action"] = "next"
        elif "previous" in text_lower or "prev" in text_lower:
            params["action"] = "previous"
        elif "last" in text_lower:
            params["action"] = "reopen"
        elif tab_match:
            params["action"] = "switch"

    elif action == IntentType.KEYBOARD:
        # Extract keyboard shortcut components
        keys = []

        # Check for modifier keys
        if "command" in text_lower or "cmd" in text_lower:
            keys.append("command")
        if "control" in text_lower or "ctrl" in text_lower:
            keys.append("control")
        if "alt" in text_lower or "option" in text_lower:
            keys.append("option")
        if "shift" in text_lower:
            keys.append("shift")

        # Map common shortcuts to key combinations
        shortcut_map = {
            "copy": ["command", "c"],
            "paste": ["command", "v"],
            "cut": ["command", "x"],
            "undo": ["command", "z"],
            "redo": ["command", "shift", "z"],
            "save": ["command", "s"],
            "select all": ["command", "a"],
            "find": ["command", "f"],
            "delete": ["backspace"],
        }

        for shortcut, shortcut_keys in shortcut_map.items():
            if shortcut in text_lower:
                params["keys"] = shortcut_keys
                break

        # If no predefined shortcut, try to extract the key
        if "keys" not in params and keys:
            # Look for a single letter after the modifier
            key_match = re.search(r'(?:command|cmd|ctrl|control|alt|option|shift)[\s+\-]([a-z0-9])', text_lower)
            if key_match:
                keys.append(key_match.group(1))
            params["keys"] = keys

        # Check for single key presses
        single_keys = ["enter", "return", "escape", "esc", "tab", "space", "backspace", "delete"]
        for key in single_keys:
            if key in text_lower and "keys" not in params:
                params["keys"] = [key]
                break

    elif action == IntentType.MEMORY:
        # Determine memory operation type
        if "remember" in text_lower or "note" in text_lower or "keep in mind" in text_lower or "forget" not in text_lower:
            params["operation"] = "store"
        elif "recall" in text_lower or "know about" in text_lower or "remember" in text_lower:
            params["operation"] = "retrieve"
        elif "forget" in text_lower:
            params["operation"] = "delete"
        elif "list" in text_lower or "show" in text_lower:
            params["operation"] = "list"

    elif action == IntentType.TYPE:
        # Extract the text to type
        type_match = re.search(r'^(?:type|write|enter|input)\s+(?:in\s+)?["\']?(.+?)["\']?$', text_lower)
        if type_match:
            params["text"] = type_match.group(1)

    return params


# =============================================================================
# MAIN PARSER FUNCTION
# =============================================================================

def parse(text: str) -> Intent:
    """Parse natural language input into a structured Intent.

    This is the main entry point for intent parsing. It normalizes the input,
    classifies the intent, extracts the target and parameters, and returns
    a fully populated Intent object.

    Args:
        text: The raw user input text.

    Returns:
        An Intent object with all parsed information.

    Example:
        >>> intent = parse("click on the Chrome icon in the dock")
        >>> intent.action
        <IntentType.CLICK: 'click'>
        >>> intent.target
        'Chrome'
        >>> intent.confidence
        1.0
    """
    if not text:
        return Intent(
            action=IntentType.UNKNOWN,
            confidence=0.0,
            raw_text=text or "",
            requires_ai=True
        )

    # Store original text
    raw_text = text

    # Clean voice input (remove conversational prefixes/suffixes)
    cleaned = clean_voice_input(text)

    # Normalize speech input (fix fragmented words)
    normalized = normalize_speech(cleaned)

    # Classify the intent
    action, confidence = _classify_intent(normalized)

    # Extract target
    target = extract_target(normalized, action)

    # Extract additional parameters
    params = _extract_params(normalized, action)

    # Determine if AI assistance is needed
    requires_ai = confidence < 0.5 or action == IntentType.UNKNOWN

    # If confidence is very low, treat as conversation
    if action == IntentType.UNKNOWN or confidence < 0.3:
        action = IntentType.CONVERSATION
        requires_ai = True

    return Intent(
        action=action,
        target=target,
        params=params,
        confidence=confidence,
        raw_text=raw_text,
        requires_ai=requires_ai
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_supported_intents() -> List[str]:
    """Get a list of all supported intent types.

    Returns:
        List of intent type names.
    """
    return [intent.value for intent in IntentType if intent != IntentType.UNKNOWN]


def get_intent_keywords(intent_type: IntentType) -> List[str]:
    """Get keywords associated with an intent type.

    Args:
        intent_type: The intent type to get keywords for.

    Returns:
        List of keywords that trigger this intent type.
    """
    return INTENT_KEYWORDS.get(intent_type, [])


def is_action_keyword(word: str) -> bool:
    """Check if a word is a recognized action keyword.

    Args:
        word: The word to check.

    Returns:
        True if the word is an action keyword.
    """
    word = word.lower()
    for keywords in INTENT_KEYWORDS.values():
        if word in keywords:
            return True
    return False
