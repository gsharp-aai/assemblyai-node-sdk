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
  // Per-channel rolling buffer of (ts, rms) for active frames only. Used by
  // `referenceRmsForChannel` to compute a P95 reference loudness over the
  // last `windowMs`, which the cross-channel suppression rule in
  // `scoreChannels` uses to normalize away gain asymmetry between channels.
  // Entries older than `windowMs` are evicted on push; ordering is FIFO so
  // eviction is a simple head-advance.
  private activeByChannel = new Map<
    string,
    { entries: Array<{ ts: number; rms: number }>; head: number }
  >();

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

    if (frame.active) {
      let bucket = this.activeByChannel.get(frame.channel);
      if (!bucket) {
        bucket = { entries: [], head: 0 };
        this.activeByChannel.set(frame.channel, bucket);
      }
      bucket.entries.push({ ts: frame.ts, rms: frame.rms });
      while (
        bucket.head < bucket.entries.length &&
        bucket.entries[bucket.head].ts < cutoff
      ) {
        bucket.head++;
      }
      if (bucket.head > 256 && bucket.head * 2 > bucket.entries.length) {
        bucket.entries = bucket.entries.slice(bucket.head);
        bucket.head = 0;
      }
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

  /**
   * P95 of recent active-frame RMS for a channel, or `null` if the channel
   * doesn't have enough active history yet for the reference to be
   * meaningful. Caller falls back to raw RMS when this returns `null`
   * (treats channels as equally-scaled until enough data accumulates).
   */
  referenceRmsForChannel(channel: string): number | null {
    const bucket = this.activeByChannel.get(channel);
    if (!bucket) return null;
    const n = bucket.entries.length - bucket.head;
    if (n < MIN_ACTIVE_FRAMES_FOR_REFERENCE) return null;
    const sorted = new Float64Array(n);
    for (let i = 0; i < n; i++) sorted[i] = bucket.entries[bucket.head + i].rms;
    sorted.sort();
    // P95 = element at index ceil(0.95 * n) - 1, clamped.
    const idx = Math.min(n - 1, Math.max(0, Math.ceil(0.95 * n) - 1));
    return sorted[idx];
  }

  clear(): void {
    this.frames = [];
    this.head = 0;
    this.activeByChannel.clear();
  }
}

/**
 * Minimum number of active frames required before a channel has a
 * meaningful P95 reference. ~600 ms at 20 ms VAD-frame cadence. Below this
 * the cross-channel suppression rule treats channels as equally-scaled
 * (no normalization applied) so brand-new sessions don't get random
 * suppression decisions.
 */
const MIN_ACTIVE_FRAMES_FOR_REFERENCE = 30;

/**
 * Per-instant cross-channel suppression threshold. At each `ts`, a channel
 * must be at least this fraction as loud (in normalized units) as the
 * loudest channel to count toward its own per-word score. Below the
 * threshold the channel's frame is treated as leakage / bleed from the
 * louder channel and contributes 0. Same numerical value as the per-word
 * `dominanceRatio` (4×), applied at the per-`ts` level.
 */
const CROSS_CHANNEL_SUPPRESSION_THRESHOLD = 0.25;

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
 * Sum per-channel `rms` over active-only frames, applying a per-`ts`
 * cross-channel suppression rule.
 *
 * At each timestamp `ts` represented in `frames`, every active channel's
 * RMS is normalized by the channel's rolling P95 active RMS (a stable
 * reference for "how loud does this channel usually get when it's active").
 * The channel with the loudest normalized energy at `ts` is treated as the
 * source; other active channels whose normalized RMS is below
 * `CROSS_CHANNEL_SUPPRESSION_THRESHOLD × loudestNorm` are treated as
 * leakage from that source and contribute 0 to their own score for that
 * `ts`.
 *
 * This is the only rule that compares channels against each other at the
 * same instant. The normalization is **only** used to make those
 * cross-channel comparisons fair across asymmetric channel gains — the
 * amount accumulated into each channel's score is still raw `rms`, not the
 * normalized value, so total scores stay in the same units that
 * `pickDominant`'s multiplicative `dominanceRatio` was tuned against.
 *
 * Channels without enough active history (< `MIN_ACTIVE_FRAMES_FOR_REFERENCE`)
 * are treated as equally-scaled — their normalized loudness is just their
 * raw RMS, which is fine because in a balanced setup raw RMS comparisons
 * already work. Suppression only kicks in once both channels have stable
 * references.
 */
function scoreChannels(
  frames: VadFrame[],
  timeline: VadTimeline,
): Map<string, number> {
  // Group frames by ts. There can be one frame per channel at each ts.
  const byTs = new Map<number, VadFrame[]>();
  for (const f of frames) {
    let bucket = byTs.get(f.ts);
    if (!bucket) {
      bucket = [];
      byTs.set(f.ts, bucket);
    }
    bucket.push(f);
  }

  const scores = new Map<string, number>();
  // Reference cache so we don't recompute P95 per channel for every `ts` in
  // the window — references change slowly relative to a single word's span.
  const refs = new Map<string, number | null>();
  const refFor = (channel: string): number | null => {
    if (!refs.has(channel)) refs.set(channel, timeline.referenceRmsForChannel(channel));
    return refs.get(channel)!;
  };

  for (const bucket of byTs.values()) {
    // Compute normalized loudness per active channel at this ts.
    let loudestNorm = 0;
    const normByChannel = new Map<string, number>();
    for (const f of bucket) {
      if (!f.active) continue;
      const ref = refFor(f.channel);
      const norm = ref !== null && ref > 0 ? f.rms / ref : f.rms;
      normByChannel.set(f.channel, norm);
      if (norm > loudestNorm) loudestNorm = norm;
    }
    if (loudestNorm <= 0) continue;

    for (const f of bucket) {
      if (!f.active) continue;
      const norm = normByChannel.get(f.channel) ?? 0;
      const ratio = norm / loudestNorm;
      if (ratio < CROSS_CHANNEL_SUPPRESSION_THRESHOLD) continue; // suppressed as leakage
      scores.set(f.channel, (scores.get(f.channel) ?? 0) + f.rms);
    }
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
  const scores = scoreChannels(framesInWin, timeline);
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
