//! OpenAI `stop` sequences.
//!
//! What is matched: the VISIBLE CONTENT only -- the text the DSML scanner
//! emits as `DsmlEvent::Text` (outside tool-call frames), after the content
//! channel's UTF-8 assembly.
//!   * Not reasoning: OpenAI applies `stop` to the generated content, and a
//!     ReAct client's `"\nObservation:"` showing up in the model's thinking
//!     must not end the turn before the answer starts.
//!   * Not inside a `<｜DSML｜tool_calls>` frame: a parameter value that
//!     contains the stop string would otherwise cut the call mid-frame, and
//!     `DsmlScanner::finish` would "repair" the truncated frame into a call
//!     with wrong arguments. The scanner never emits frame bytes as `Text`, so
//!     matching its `Text` output is exactly this rule; a tool-call event
//!     flushes the held-back tail ([`StopMatcher::flush`]) because the text on
//!     the far side of a frame is not contiguous with it.
//!
//! On a match the content is cut BEFORE the stop string, the handler flips the
//! request's cancel flag -- the path a client disconnect takes: multistream
//! drops the stream and releases its arena slot at the next tick exactly as
//! `finish` does, the serial worker drops its live cache (whose KV now holds
//! tokens past the stop string the client will never replay) -- and the turn
//! ends with `finish_reason: "stop"`.

use crate::openai::types::StopSpec;

/// Most `stop` entries served. OpenAI's own limit is 4, but vLLM-style
/// clients send more and the matcher handles any count; this only bounds
/// its per-chunk work.
pub const MAX_STOP_SEQUENCES: usize = 64;

/// The request's stop strings, validated. More than [`MAX_STOP_SEQUENCES`] is
/// a 400 (the message names the parameter). An empty string is dropped: it
/// would "match" before the first byte and end every turn with no content.
pub fn stop_sequences(spec: Option<&StopSpec>) -> Result<Vec<String>, String> {
    let v = spec.map(StopSpec::to_vec).unwrap_or_default();
    if v.len() > MAX_STOP_SEQUENCES {
        return Err(format!(
            "stop: at most {MAX_STOP_SEQUENCES} stop sequences are supported, got {}",
            v.len()
        ));
    }
    Ok(v.into_iter().filter(|s| !s.is_empty()).collect())
}

/// Incremental stop-string matcher over a text stream.
///
/// [`push`](Self::push) returns the text that is safe to emit now. It holds
/// back the longest tail that is a proper prefix of some stop string (so at
/// most `longest - 1` bytes), which is what keeps a stop string split across
/// chunks from ever reaching the client. Once a stop string matched,
/// [`hit`](Self::hit) is true and everything after it is discarded.
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    /// Longest stop string minus one: the most that can be held back.
    max_hold: usize,
    held: String,
    hit: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        let max_hold = stops.iter().map(|s| s.len()).max().unwrap_or(0).saturating_sub(1);
        StopMatcher { stops, max_hold, held: String::new(), hit: false }
    }

    pub fn hit(&self) -> bool {
        self.hit
    }

    /// Feed the next piece of visible content; returns what may be emitted.
    pub fn push(&mut self, s: &str) -> String {
        if self.hit {
            return String::new();
        }
        if self.stops.is_empty() {
            return s.to_owned();
        }
        self.held.push_str(s);
        // Earliest match wins. `held` never contains a whole stop string from
        // an earlier push (it would have matched then), so a search over it is
        // complete.
        if let Some(at) = self.stops.iter().filter_map(|t| self.held.find(t.as_str())).min() {
            self.hit = true;
            let mut out = std::mem::take(&mut self.held);
            out.truncate(at);
            return out;
        }
        let keep = self.partial_tail_len();
        let tail = self.held.split_off(self.held.len() - keep);
        std::mem::replace(&mut self.held, tail)
    }

    /// Release the held-back tail: nothing can complete a match any more (end
    /// of output, or a tool-call frame follows).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Length of the longest tail of `held` that is a proper prefix of a stop
    /// string. Always a char boundary: the tail's first byte equals a stop
    /// string's first byte, which starts a char.
    fn partial_tail_len(&self) -> usize {
        let h = self.held.as_bytes();
        (1..=self.max_hold.min(h.len()))
            .rev()
            .find(|&k| self.stops.iter().any(|t| t.len() > k && t.as_bytes().starts_with(&h[h.len() - k..])))
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(stops: &[&str]) -> StopMatcher {
        StopMatcher::new(stops.iter().map(|s| s.to_string()).collect())
    }

    /// Feed `chunks`; return (everything emitted incl. the final flush, hit).
    fn run(stops: &[&str], chunks: &[&str]) -> (String, bool) {
        let mut sm = m(stops);
        let mut out = String::new();
        for c in chunks {
            out.push_str(&sm.push(c));
            if sm.hit() {
                return (out, true);
            }
        }
        out.push_str(&sm.flush());
        (out, false)
    }

    #[test]
    fn no_stops_is_a_pass_through() {
        let mut sm = m(&[]);
        assert_eq!(sm.push("abc"), "abc");
        assert_eq!(sm.flush(), "");
        assert!(!sm.hit());
    }

    #[test]
    fn match_inside_one_chunk_truncates_before_the_stop() {
        assert_eq!(run(&["\nObservation:"], &["Thought: x\nObservation: y"]), ("Thought: x".into(), true));
        // The stop at the very start: empty content, still a hit.
        assert_eq!(run(&["END"], &["END and more"]), (String::new(), true));
    }

    #[test]
    fn match_split_across_chunks_is_never_emitted() {
        let mut sm = m(&["\nObservation:"]);
        assert_eq!(sm.push("Action: ls\n"), "Action: ls");
        assert_eq!(sm.push("Obser"), "", "a growing partial match stays held");
        assert_eq!(sm.push("vation"), "");
        assert_eq!(sm.push(": result"), "");
        assert!(sm.hit());
        assert_eq!(sm.push("more"), "", "nothing after a hit");
        assert_eq!(sm.flush(), "");
        // One byte per chunk.
        let text = "abc STOP def";
        let chunks: Vec<String> = text.chars().map(String::from).collect();
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        assert_eq!(run(&["STOP"], &refs), ("abc ".into(), true));
    }

    #[test]
    fn a_partial_that_does_not_complete_is_released() {
        let mut sm = m(&["STOP"]);
        assert_eq!(sm.push("ab ST"), "ab ");
        assert_eq!(sm.push("ART"), "START", "the held 'ST' goes out once it cannot match");
        assert_eq!(sm.push("x ST"), "x ");
        assert_eq!(sm.flush(), "ST", "end of output releases the tail");
        assert!(!sm.hit());
    }

    #[test]
    fn holdback_is_at_most_longest_minus_one_and_only_a_real_prefix() {
        let mut sm = m(&["abcd", "xy"]);
        // "abc" is a 3-byte prefix of "abcd": all held (= longest - 1).
        assert_eq!(sm.push("zzabc"), "zz");
        assert_eq!(sm.flush(), "abc");
        // Tail "q" prefixes nothing: nothing held.
        assert_eq!(sm.push("abq"), "abq");
        // Tail "x" prefixes "xy" only.
        assert_eq!(sm.push("--x"), "--");
        assert_eq!(sm.push("z"), "xz");
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        assert_eq!(run(&["world", "lo"], &["hello world"]), ("hel".into(), true));
        // Split: "lo" completes in the second chunk before "world" could.
        assert_eq!(run(&["world", "lo w"], &["hel", "lo wor", "ld"]), ("hel".into(), true));
    }

    #[test]
    fn multibyte_stop_and_text_keep_char_boundaries() {
        // Stop string with a multi-byte char; the text around it is
        // multi-byte too. split_off/truncate would panic off a boundary.
        assert_eq!(run(&["──end"], &["héllo ─", "─e", "nd tail"]), ("héllo ".into(), true));
        let mut sm = m(&["日本"]);
        assert_eq!(sm.push("こんにちは日"), "こんにちは");
        assert_eq!(sm.push("曜"), "日曜");
    }

    #[test]
    fn stop_field_validation() {
        assert_eq!(stop_sequences(None), Ok(vec![]));
        assert_eq!(stop_sequences(Some(&StopSpec::One("x".into()))), Ok(vec!["x".to_string()]));
        let four = StopSpec::Many(vec!["a".into(), "b".into(), "".into(), "d".into()]);
        assert_eq!(stop_sequences(Some(&four)), Ok(vec!["a".to_string(), "b".into(), "d".into()]), "empty strings dropped");
        let five = StopSpec::Many((0..5).map(|i| i.to_string()).collect());
        assert_eq!(stop_sequences(Some(&five)).map(|v| v.len()), Ok(5), "more than OpenAI's 4 are served");
        let many = StopSpec::Many((0..=MAX_STOP_SEQUENCES).map(|i| i.to_string()).collect());
        let e = stop_sequences(Some(&many)).unwrap_err();
        assert!(e.starts_with("stop:"), "{e}");
    }
}
