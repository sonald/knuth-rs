//! Server-Sent-Events parser. No 1:1 TS counterpart — the TS package depends on the third-party
//! `eventsource-parser`. We inline a minimal parser here so providers don't have to.
//!
//! Spec: <https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation>
//! We deliberately implement only what providers use (`event:` and `data:` fields; comments
//! starting with `:` are dropped).

use bytes::{Bytes, BytesMut};
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

#[derive(Clone, Debug, Default)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Stream adapter: turns an HTTP body byte stream into a stream of parsed `SseEvent`s.
pub struct SseStream<S> {
    inner: S,
    buf: BytesMut,
    current: SseEvent,
    pending: Vec<SseEvent>,
    upstream_done: bool,
}

impl<S> SseStream<S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buf: BytesMut::new(),
            current: SseEvent::default(),
            pending: Vec::new(),
            upstream_done: false,
        }
    }

    fn flush_current(&mut self) {
        if !self.current.data.is_empty() || self.current.event.is_some() {
            self.pending.push(std::mem::take(&mut self.current));
        }
    }

    fn drain_lines(&mut self) {
        // Find each complete line in `self.buf`.
        loop {
            let nl = match self.buf.iter().position(|&b| b == b'\n') {
                Some(i) => i,
                None => return,
            };
            let line = self.buf.split_to(nl + 1);
            let line = std::str::from_utf8(&line[..line.len().saturating_sub(1)])
                .unwrap_or("")
                .trim_end_matches('\r');
            if line.is_empty() {
                self.flush_current();
                continue;
            }
            if line.starts_with(':') {
                // Comment — ignore.
                continue;
            }
            let (field, raw) = match line.find(':') {
                Some(i) => (&line[..i], &line[i + 1..]),
                None => (line, ""),
            };
            match field {
                "event" => self.current.event = Some(raw.trim_start_matches(' ').to_string()),
                "data" => {
                    if !self.current.data.is_empty() {
                        self.current.data.push('\n');
                    }
                    self.current
                        .data
                        .push_str(raw.strip_prefix(' ').unwrap_or(raw));
                }
                _ => {} // ignore `id`, `retry`, etc.
            }
        }
    }
}

impl<S> Stream for SseStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<SseEvent, reqwest::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(ev) = self.pending.pop() {
                return Poll::Ready(Some(Ok(ev)));
            }
            if self.upstream_done {
                self.flush_current();
                if let Some(ev) = self.pending.pop() {
                    return Poll::Ready(Some(Ok(ev)));
                }
                return Poll::Ready(None);
            }
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    if !self.buf.is_empty() {
                        self.buf.extend_from_slice(b"\n");
                        self.drain_lines();
                        self.pending.reverse();
                    }
                    self.upstream_done = true;
                }
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buf.extend_from_slice(&chunk);
                    self.drain_lines();
                    // Order: FIFO. drain_lines pushes to back; we pop from back. Reverse so the
                    // first event we parsed comes out first.
                    self.pending.reverse();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{TryStreamExt, stream};

    async fn parse(input: &'static [u8]) -> Vec<SseEvent> {
        parse_chunks(&[input]).await
    }

    async fn parse_chunks(chunks: &[&'static [u8]]) -> Vec<SseEvent> {
        SseStream::new(stream::iter(
            chunks
                .iter()
                .map(|chunk| Ok::<_, reqwest::Error>(Bytes::from_static(chunk))),
        ))
        .try_collect()
        .await
        .expect("SSE input is valid")
    }

    #[tokio::test]
    async fn data_field_removes_only_one_leading_space() {
        let cases = [
            (b"data: one\ndata: two\n\n".as_slice(), vec!["one\ntwo"]),
            (b"data:no-space\n\n".as_slice(), vec!["no-space"]),
            (
                b"data:  two-leading-spaces\n\n".as_slice(),
                vec![" two-leading-spaces"],
            ),
            (b"data:\n\n".as_slice(), vec![]),
            (b"data: crlf\r\n\r\n".as_slice(), vec!["crlf"]),
        ];

        for (input, expected) in cases {
            let events = parse(input).await;
            assert_eq!(
                events
                    .into_iter()
                    .map(|event| event.data)
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[tokio::test]
    async fn multiple_events_are_emitted_fifo() {
        let events = parse(b"event: first\ndata: one\n\nevent: second\ndata: two\n\n").await;

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("first"));
        assert_eq!(events[0].data, "one");
        assert_eq!(events[1].event.as_deref(), Some("second"));
        assert_eq!(events[1].data, "two");
    }

    #[tokio::test]
    async fn line_and_data_split_across_chunks_are_reassembled() {
        let events = parse_chunks(&[b"eve", b"nt: split\ndata: hel", b"lo\ndata: world\n\n"]).await;

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("split"));
        assert_eq!(events[0].data, "hello\nworld");
    }

    #[tokio::test]
    async fn eof_without_final_line_feed_flushes_current_event() {
        let events = parse(b"event: final\ndata: done").await;

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("final"));
        assert_eq!(events[0].data, "done");
    }
}
