"""
Skill Recorder

Records user demonstrations and converts them to replayable skills.
Captures actions, visual context, and timing for adaptive replay.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

from .types import (
    ActionType,
    RecordedAction,
    LearnedSkill,
    RecordingSession,
    VisualTarget,
)


class SkillRecorder:
    """
    Records user demonstrations and converts them to learned skills.

    Usage:
        recorder = SkillRecorder(storage_path)

        # Start recording
        session = recorder.start_recording("book_flight")

        # Record actions as they happen
        recorder.record_click(500, 300, visual_description="Search button")
        recorder.record_type("NYC to LAX")
        recorder.record_hotkey(["command", "enter"])

        # Stop and save
        skill = recorder.stop_recording(
            trigger_phrases=["book a flight", "search flights"]
        )
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the skill recorder.

        Args:
            storage_path: Where to persist learned skills
        """
        self.storage_path = storage_path or Path.home() / ".aria" / "learned_skills"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.current_session: Optional[RecordingSession] = None
        self.learned_skills: Dict[str, LearnedSkill] = {}

        # Callbacks
        self.on_action_recorded: Optional[Callable[[RecordedAction], None]] = None
        self.on_recording_started: Optional[Callable[[RecordingSession], None]] = None
        self.on_recording_stopped: Optional[Callable[[LearnedSkill], None]] = None

        # Load existing skills
        self._load_skills()

    def _load_skills(self) -> None:
        """Load learned skills from storage."""
        skills_file = self.storage_path / "skills.json"
        if skills_file.exists():
            try:
                with open(skills_file) as f:
                    data = json.load(f)
                for skill_data in data.get("skills", []):
                    skill = LearnedSkill.from_dict(skill_data)
                    self.learned_skills[skill.id] = skill
                print(f"Loaded {len(self.learned_skills)} learned skills")
            except Exception as e:
                print(f"Error loading skills: {e}")

    def _save_skills(self) -> None:
        """Persist learned skills to storage."""
        skills_file = self.storage_path / "skills.json"
        try:
            data = {
                "skills": [s.to_dict() for s in self.learned_skills.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(skills_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving skills: {e}")

    # =========================================================================
    # Recording Session Management
    # =========================================================================

    def start_recording(
        self,
        name: str,
        starting_app: Optional[str] = None,
        starting_url: Optional[str] = None,
    ) -> RecordingSession:
        """
        Start a new recording session.

        Args:
            name: Name for this skill (e.g., "book_flight")
            starting_app: Current app when recording starts
            starting_url: Current URL if in browser

        Returns:
            The new recording session
        """
        if self.current_session:
            print(f"Warning: Stopping existing recording '{self.current_session.name}'")
            self.cancel_recording()

        self.current_session = RecordingSession(
            id=str(uuid.uuid4())[:8],
            name=name,
            starting_app=starting_app,
            starting_url=starting_url,
        )

        if self.on_recording_started:
            self.on_recording_started(self.current_session)

        print(f"Started recording skill: {name}")
        return self.current_session

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.current_session is not None and not self.current_session.is_paused

    def pause_recording(self) -> None:
        """Pause the current recording."""
        if self.current_session:
            self.current_session.is_paused = True
            print("Recording paused")

    def resume_recording(self) -> None:
        """Resume a paused recording."""
        if self.current_session:
            self.current_session.is_paused = False
            print("Recording resumed")

    def cancel_recording(self) -> None:
        """Cancel the current recording without saving."""
        if self.current_session:
            print(f"Cancelled recording: {self.current_session.name}")
            self.current_session = None

    def stop_recording(
        self,
        trigger_phrases: Optional[List[str]] = None,
        description: Optional[str] = None,
        success_criteria: Optional[str] = None,
    ) -> Optional[LearnedSkill]:
        """
        Stop recording and convert to a learned skill.

        Args:
            trigger_phrases: Phrases that trigger this skill
            description: Human-readable description
            success_criteria: How to verify the skill worked

        Returns:
            The learned skill, or None if recording was empty
        """
        if not self.current_session:
            print("No active recording to stop")
            return None

        if not self.current_session.actions:
            print("Recording is empty, nothing to save")
            self.current_session = None
            return None

        # Create the skill from the recording
        skill = self._process_recording_to_skill(
            trigger_phrases=trigger_phrases or [self.current_session.name],
            description=description,
            success_criteria=success_criteria,
        )

        # Save and cleanup
        self.learned_skills[skill.id] = skill
        self._save_skills()

        if self.on_recording_stopped:
            self.on_recording_stopped(skill)

        print(f"Saved skill '{skill.name}' with {len(skill.actions)} actions")
        self.current_session = None
        return skill

    def _process_recording_to_skill(
        self,
        trigger_phrases: List[str],
        description: Optional[str],
        success_criteria: Optional[str],
    ) -> LearnedSkill:
        """Convert a recording session to a learned skill."""
        session = self.current_session

        # Find decision points (actions marked by user or with significant delays)
        decision_points = []
        for i, action in enumerate(session.actions):
            if action.is_decision_point or action.delay_before_ms > 3000:
                decision_points.append(i)

        # Generate description if not provided
        if not description:
            action_types = [a.action_type.value for a in session.actions[:5]]
            description = f"Learned skill with {len(session.actions)} actions: {', '.join(action_types)}"

        return LearnedSkill(
            id=str(uuid.uuid4())[:8],
            name=session.name,
            description=description,
            trigger_phrases=trigger_phrases,
            actions=session.actions.copy(),
            decision_points=decision_points,
            success_criteria=success_criteria,
            required_app=session.starting_app,
            required_url_pattern=session.starting_url,
            learned_from_user=True,
            confidence=0.6,  # Start with moderate confidence
        )

    # =========================================================================
    # Recording Actions
    # =========================================================================

    def record_action(self, action: RecordedAction) -> None:
        """
        Record a generic action.

        Args:
            action: The action to record
        """
        if not self.current_session or self.current_session.is_paused:
            return

        # Calculate delay since last action
        if self.current_session.last_action_time:
            delta = datetime.now() - self.current_session.last_action_time
            action.delay_before_ms = int(delta.total_seconds() * 1000)

        self.current_session.actions.append(action)
        self.current_session.last_action_time = datetime.now()

        if self.on_action_recorded:
            self.on_action_recorded(action)

    def record_click(
        self,
        x: int,
        y: int,
        visual_description: Optional[str] = None,
        element_text: Optional[str] = None,
        element_type: Optional[str] = None,
        screenshot_region: Optional[str] = None,
        app_name: Optional[str] = None,
        is_decision_point: bool = False,
    ) -> None:
        """
        Record a click action.

        Args:
            x, y: Click coordinates
            visual_description: What the clicked element looks like
            element_text: Text on the element
            element_type: Type of element (button, link, etc.)
            screenshot_region: Base64 of area around click
            app_name: Current application
            is_decision_point: Whether this is a key decision
        """
        visual_target = None
        if visual_description or element_text or screenshot_region:
            visual_target = VisualTarget(
                description=visual_description or "",
                element_text=element_text,
                element_type=element_type,
                screenshot_region=screenshot_region,
            )

        action = RecordedAction(
            action_type=ActionType.CLICK,
            x=x,
            y=y,
            visual_target=visual_target,
            app_name=app_name,
            is_decision_point=is_decision_point,
        )
        self.record_action(action)

    def record_double_click(
        self,
        x: int,
        y: int,
        visual_description: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> None:
        """Record a double-click action."""
        visual_target = None
        if visual_description:
            visual_target = VisualTarget(description=visual_description)

        action = RecordedAction(
            action_type=ActionType.DOUBLE_CLICK,
            x=x,
            y=y,
            visual_target=visual_target,
            app_name=app_name,
        )
        self.record_action(action)

    def record_type(
        self,
        text: str,
        is_variable: bool = False,
        variable_name: Optional[str] = None,
    ) -> None:
        """
        Record a typing action.

        Args:
            text: The text that was typed
            is_variable: Whether this should be variable (filled in at runtime)
            variable_name: Name of variable if is_variable
        """
        action = RecordedAction(
            action_type=ActionType.TYPE,
            text=text,
            notes=f"variable:{variable_name}" if is_variable else None,
        )
        self.record_action(action)

    def record_scroll(
        self,
        amount: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> None:
        """
        Record a scroll action.

        Args:
            amount: Scroll amount (positive=up, negative=down)
            x, y: Position to scroll at (optional)
        """
        action = RecordedAction(
            action_type=ActionType.SCROLL,
            scroll_amount=amount,
            x=x,
            y=y,
        )
        self.record_action(action)

    def record_hotkey(
        self,
        keys: List[str],
        description: Optional[str] = None,
    ) -> None:
        """
        Record a keyboard shortcut.

        Args:
            keys: The keys pressed together (e.g., ["command", "c"])
            description: What this shortcut does
        """
        action = RecordedAction(
            action_type=ActionType.HOTKEY,
            keys=keys,
            notes=description,
        )
        self.record_action(action)

    def record_key_press(
        self,
        key: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Record a single key press.

        Args:
            key: The key pressed (e.g., "enter", "tab")
            description: What this key does in context
        """
        action = RecordedAction(
            action_type=ActionType.KEY_PRESS,
            keys=[key],
            notes=description,
        )
        self.record_action(action)

    def record_wait(
        self,
        milliseconds: int,
        reason: Optional[str] = None,
    ) -> None:
        """
        Record an explicit wait.

        Args:
            milliseconds: How long to wait
            reason: Why we're waiting (e.g., "page loading")
        """
        action = RecordedAction(
            action_type=ActionType.WAIT,
            delay_before_ms=milliseconds,
            notes=reason,
        )
        self.record_action(action)

    def record_open_app(
        self,
        app_name: str,
    ) -> None:
        """Record opening an application."""
        action = RecordedAction(
            action_type=ActionType.OPEN_APP,
            app_name=app_name,
        )
        self.record_action(action)

    def record_open_url(
        self,
        url: str,
    ) -> None:
        """Record opening a URL."""
        action = RecordedAction(
            action_type=ActionType.OPEN_URL,
            text=url,
        )
        self.record_action(action)

    def mark_decision_point(
        self,
        description: str,
    ) -> None:
        """
        Mark the last action as a decision point.

        Decision points are places where the user made a choice
        and might need to be prompted during replay.

        Args:
            description: Why this is a decision point
        """
        if self.current_session and self.current_session.actions:
            last_action = self.current_session.actions[-1]
            last_action.is_decision_point = True
            last_action.notes = (last_action.notes or "") + f" [Decision: {description}]"

    def add_note(self, note: str) -> None:
        """Add a note to the current recording."""
        if self.current_session:
            self.current_session.notes.append(note)

    # =========================================================================
    # Skill Management
    # =========================================================================

    def get_skill(self, skill_id: str) -> Optional[LearnedSkill]:
        """Get a skill by ID."""
        return self.learned_skills.get(skill_id)

    def get_skill_by_name(self, name: str) -> Optional[LearnedSkill]:
        """Get a skill by name."""
        for skill in self.learned_skills.values():
            if skill.name.lower() == name.lower():
                return skill
        return None

    def find_skill_by_trigger(self, text: str) -> Optional[LearnedSkill]:
        """
        Find a skill that matches the given trigger text.

        Args:
            text: User input that might trigger a skill

        Returns:
            Best matching skill, or None
        """
        text_lower = text.lower()
        best_match: Optional[LearnedSkill] = None
        best_score = 0.0

        for skill in self.learned_skills.values():
            if skill.is_archived if hasattr(skill, 'is_archived') else False:
                continue

            for trigger in skill.trigger_phrases:
                if trigger.lower() in text_lower:
                    # Exact phrase match
                    score = len(trigger) / len(text)
                    if score > best_score:
                        best_score = score
                        best_match = skill

            # Also check name
            if skill.name.lower() in text_lower:
                score = len(skill.name) / len(text) * 0.8  # Slightly lower priority
                if score > best_score:
                    best_score = score
                    best_match = skill

        return best_match if best_score > 0.3 else None

    def list_skills(self) -> List[LearnedSkill]:
        """Get all learned skills."""
        return list(self.learned_skills.values())

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill by ID."""
        if skill_id in self.learned_skills:
            del self.learned_skills[skill_id]
            self._save_skills()
            return True
        return False

    def update_skill_triggers(
        self,
        skill_id: str,
        trigger_phrases: List[str],
    ) -> bool:
        """Update a skill's trigger phrases."""
        skill = self.learned_skills.get(skill_id)
        if skill:
            skill.trigger_phrases = trigger_phrases
            skill.updated_at = datetime.now()
            self._save_skills()
            return True
        return False


# Singleton instance
_recorder: Optional[SkillRecorder] = None


def get_skill_recorder() -> SkillRecorder:
    """Get the singleton SkillRecorder instance."""
    global _recorder
    if _recorder is None:
        _recorder = SkillRecorder()
    return _recorder
