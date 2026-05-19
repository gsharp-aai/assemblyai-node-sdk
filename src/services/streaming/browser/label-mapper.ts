import { StreamingWord, TurnEvent } from "../../../types/streaming";
import { Channel, VadFrame } from "../../../types/streaming/dual-channel";

export type LabelMapperParams = {
  /** Per-word energy ratio above which a channel is declared dominant. */
  dominanceRatio: number;
};

/**
 * Append-only ring buffer of VAD frames in stream-relative ms order.
 * `pushFrame` is O(1) amortized; `framesInWindow` is O(n) over kept frames,
 * which is fine for the per-word lookups we do (a 30 s window at 50 frames/s
 * per channel × 2 channels = 3000 entries, scanned once per word).
 *
 * Runtime-agnostic — no DOM or Web Audio dependencies. Lives under `browser/`
 * only for import-path stability with earlier releases.
 */
export class VadTimeline {
  private frames: VadFrame[] = [];
  private head = 0;

  constructor(private readonly windowMs: number) {}

  pushFrame(frame: VadFrame): void {
    this.frames.push(frame);
    const cutoff = frame.ts - this.windowMs;
    while (
      this.head < this.frames.length &&
      this.frames[this.head].ts < cutoff
    ) {
      this.head++;
    }
    if (this.head > 1024 && this.head * 2 > this.frames.length) {
      this.frames = this.frames.slice(this.head);
      this.head = 0;
    }
  }

  framesInWindow(startMs: number, endMs: number): VadFrame[] {
    const out: VadFrame[] = [];
    for (let i = this.head; i < this.frames.length; i++) {
      const f = this.frames[i];
      if (f.ts < startMs) continue;
      if (f.ts > endMs) break;
      out.push(f);
    }
    return out;
  }

  clear(): void {
    this.frames = [];
    this.head = 0;
  }
}

/**
 * Pad each side of the word's `[start, end]` window by this many ms when
 * scoring VAD frames. Two reasons:
 *  - Short words (e.g. 16-33 ms function words) span only 1-2 VAD frames at
 *    20 ms cadence, so missing one frame on one channel skews scoring badly.
 *  - Per-channel VAD frames can be pushed to the timeline slightly later than
 *    the streaming server emits its corresponding Turn message, leaving the
 *    exact word window short some frames at attribution time. Padding pulls
 *    in surrounding frames that are almost always from the same speaker.
 *
 * 50 ms = ~2 extra frames each side, enough to bridge per-channel `sendAudio`
 * ordering jitter without reaching far into neighboring-utterance silence.
 */
const ATTRIBUTION_WINDOW_PAD_MS = 50;

/**
 * Sum per-channel `rms` over **active-only** frames. Channels with no active
 * frames in the window are omitted (their absence is what triggers the
 * `"unknown"` return path in `attributeWord`).
 *
 * Active-only matters: a quiet-but-noisy ambient channel can have non-zero
 * raw RMS over a word window even when nothing was actually spoken there;
 * gating on `active` keeps that ambient noise from outscoring genuine speech
 * on the other channel.
 */
function scoreChannels(frames: VadFrame[]): Map<string, number> {
  const scores = new Map<string, number>();
  for (const f of frames) {
    if (!f.active) continue;
    scores.set(f.channel, (scores.get(f.channel) ?? 0) + f.rms);
  }
  return scores;
}

/**
 * Decide which channel was dominant during a word's window. Returns the top
 * channel if it beats the runner-up by at least `dominanceRatio`; otherwise
 * `null`. Empty / single-entry maps are handled explicitly so a single
 * active channel always wins outright.
 */
function pickDominant(
  scores: Map<string, number>,
  dominanceRatio: number,
): Channel | null {
  if (scores.size === 0) return null;
  const sorted = [...scores.entries()].sort((a, b) => b[1] - a[1]);
  if (sorted.length === 1) return sorted[0][0];
  const [topName, topScore] = sorted[0];
  const [, runnerScore] = sorted[1];
  if (topScore >= dominanceRatio * runnerScore) return topName;
  return null;
}

/**
 * Decide which channel was dominant during a word's `[start, end]` window.
 * Scores active VAD frames in the padded window; the top-scoring channel
 * wins if it beats the runner-up by at least `dominanceRatio`, else returns
 * `"unknown"` so the downstream window resolver can fill the word from
 * neighbor context.
 */
export function attributeWord(
  word: StreamingWord,
  timeline: VadTimeline,
  params: LabelMapperParams,
): Channel {
  const framesInWin = timeline.framesInWindow(
    word.start - ATTRIBUTION_WINDOW_PAD_MS,
    word.end + ATTRIBUTION_WINDOW_PAD_MS,
  );
  const scores = scoreChannels(framesInWin);
  return pickDominant(scores, params.dominanceRatio) ?? "unknown";
}

/**
 * Duration-weighted majority of word channels. `"unknown"` if there are no
 * words, every word resolved to `"unknown"`, or two channels tie exactly.
 */
export function rollUpTurnChannel(words: StreamingWord[]): Channel {
  const totals = new Map<string, number>();
  for (const w of words) {
    if (!w.channel || w.channel === "unknown") continue;
    const dur = Math.max(0, w.end - w.start);
    totals.set(w.channel, (totals.get(w.channel) ?? 0) + dur);
  }
  if (totals.size === 0) return "unknown";
  const sorted = [...totals.entries()].sort((a, b) => b[1] - a[1]);
  if (sorted.length === 1) return sorted[0][0];
  const [topName, topMs] = sorted[0];
  const [, runnerMs] = sorted[1];
  if (topMs === runnerMs) return "unknown";
  return topName;
}

/**
 * Mutate `turn` in place: write `turn.words[i].channel` for every word and set
 * `turn.channel` to the duration-weighted rollup.
 *
 * Returns `void` because the transcriber owns the `TurnEvent` ref and forwards
 * the same object to the customer listener — no need to allocate a copy.
 */
export function attributeTurn(
  turn: TurnEvent,
  timeline: VadTimeline,
  params: LabelMapperParams,
): void {
  for (const w of turn.words) {
    w.channel = attributeWord(w, timeline, params);
  }
  turn.channel = rollUpTurnChannel(turn.words);
}
