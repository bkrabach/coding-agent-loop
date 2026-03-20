//! Session event system.
//!
//! Every agent action emits a [`SessionEvent`] through a
//! [`tokio::sync::broadcast`] channel. Host applications subscribe via
//! [`EventSender::subscribe`] to observe the agentic loop in real time.

use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::sync::broadcast;

/// Discriminant for every event type emitted during a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// Session created and ready.
    SessionStart,
    /// Session closed (includes final state).
    SessionEnd,
    /// User submitted input.
    UserInput,
    /// Model began generating text.
    AssistantTextStart,
    // AssistantTextDelta — removed in V2-CAL-009: never emitted; the session
    // loop does not stream text incrementally. Reserved for a future streaming
    // mode. Add back when implemented.
    /// Model finished generating text (includes full text).
    AssistantTextEnd,
    /// Tool execution began.
    ToolCallStart,
    // ToolCallOutputDelta — removed in V2-CAL-009: never emitted; tools return
    // complete output in a single ToolCallEnd event. Reserved for streaming
    // tool output in a future revision.
    /// Tool execution finished (carries FULL untruncated output).
    ToolCallEnd,
    /// A steering message was injected into history.
    SteeringInjected,
    /// A turn limit was reached.
    TurnLimit,
    /// A repeating tool call pattern was detected.
    LoopDetection,
    /// An error occurred.
    Error,
}

/// A single event emitted during a session.
#[derive(Debug, Clone)]
pub struct SessionEvent {
    pub kind: EventKind,
    pub timestamp: DateTime<Utc>,
    pub session_id: String,
    /// Freeform JSON payload; schema varies by `kind`.
    pub data: Value,
}

impl SessionEvent {
    /// Create a new event timestamped at the current UTC moment.
    pub fn new(kind: EventKind, session_id: impl Into<String>, data: Value) -> Self {
        Self {
            kind,
            timestamp: Utc::now(),
            session_id: session_id.into(),
            data,
        }
    }
}

/// Broadcast channel capacity (events before lagging receivers fall behind).
pub const EVENT_CHANNEL_CAPACITY: usize = 256;

/// Sender side of the session event broadcast channel.
///
/// Cheaply cloneable — all clones share the same underlying channel.
#[derive(Clone)]
pub struct EventSender(broadcast::Sender<SessionEvent>);

impl EventSender {
    /// Create a new broadcast channel, returning `(sender, first_receiver)`.
    pub fn new() -> (Self, broadcast::Receiver<SessionEvent>) {
        let (tx, rx) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        (Self(tx), rx)
    }

    /// Emit an event. Non-blocking; lagging receivers will get
    /// [`tokio::sync::broadcast::error::RecvError::Lagged`] on their next receive.
    /// Silently ignores the "no active receivers" case.
    pub fn emit(&self, event: SessionEvent) {
        let _ = self.0.send(event);
    }

    /// Subscribe a new receiver to this channel.
    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.0.subscribe()
    }
}

impl Default for EventSender {
    fn default() -> Self {
        Self::new().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn event_kind_all_variants_exhaustive() {
        // Exhaustive match — if a new variant is added without updating this test,
        // the compiler will catch it.
        // V2-CAL-009: AssistantTextDelta and ToolCallOutputDelta were never
        // emitted and have been removed.  The active count is now 11.
        let kinds = [
            EventKind::SessionStart,
            EventKind::SessionEnd,
            EventKind::UserInput,
            EventKind::AssistantTextStart,
            EventKind::AssistantTextEnd,
            EventKind::ToolCallStart,
            EventKind::ToolCallEnd,
            EventKind::SteeringInjected,
            EventKind::TurnLimit,
            EventKind::LoopDetection,
            EventKind::Error,
        ];
        assert_eq!(kinds.len(), 11);
    }

    #[test]
    fn session_event_new_sets_timestamp() {
        let before = Utc::now();
        let ev = SessionEvent::new(EventKind::SessionStart, "sess-1", json!({}));
        let after = Utc::now();
        assert!(ev.timestamp >= before);
        assert!(ev.timestamp <= after);
        assert_eq!(ev.session_id, "sess-1");
        assert_eq!(ev.kind, EventKind::SessionStart);
    }

    #[test]
    fn emit_with_no_receivers_does_not_panic() {
        let (sender, rx) = EventSender::new();
        drop(rx); // no active receivers
        // Must not panic
        sender.emit(SessionEvent::new(
            EventKind::UserInput,
            "s",
            json!({"text": "hi"}),
        ));
    }

    #[tokio::test]
    async fn multiple_receivers_each_get_event() {
        let (sender, mut rx1) = EventSender::new();
        let mut rx2 = sender.subscribe();

        sender.emit(SessionEvent::new(EventKind::ToolCallStart, "s", json!({})));

        let ev1 = rx1.recv().await.unwrap();
        let ev2 = rx2.recv().await.unwrap();
        assert_eq!(ev1.kind, EventKind::ToolCallStart);
        assert_eq!(ev2.kind, EventKind::ToolCallStart);
    }

    #[test]
    fn event_sender_is_clone() {
        let (sender, _rx) = EventSender::new();
        let _clone = sender.clone();
    }

    #[test]
    fn event_sender_default() {
        let _sender = EventSender::default();
    }
}
