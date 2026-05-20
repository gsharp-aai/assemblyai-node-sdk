import {
  VadDetector,
  VadDetectorResult,
} from "../../../types/streaming/dual-channel";

export type EnergyVadParams = {
  /** Threshold = noiseFloor * thresholdRatio. Default 12 — picked via bench
   * sweep to tolerate moderate acoustic leak (e.g. speakers → mic) while
   * still firing reliably on real headset/handset mic speech. The original
   * 3.0 default was too sensitive: in the speaker-leak regime where mic
   * leakage is at near-system loudness, a 3× threshold trips constantly
   * and the mic VAD has no way to distinguish real speech from leak. 12
   * was empirically the lowest value that gets the speaker-leak fixture
   * to 0 mic false-positives without regressing AMI's two-headset
   * accuracy. */
  thresholdRatio?: number;
  /** EMA smoothing for the noise-floor estimate when frame is non-speech.
   * Default 0.05. */
  noiseFloorAlpha?: number;
  /** Hangover in frames: stay "active" this many frames after the last
   * speech frame. Default 3 (~60 ms at 20 ms frames). Empirically 3 is
   * tight enough that the mic VAD recovers quickly after spurious leak
   * frames, while still bridging the natural between-syllable dips of
   * real speech that 0 hangover would split. */
  hangoverFrames?: number;
  /** Initial noise floor estimate. Default 1e-4. Adaptive after the first
   * non-speech frame. */
  initialNoiseFloor?: number;
};

/**
 * Energy-based VAD with adaptive noise-floor tracking and hangover. Pure JS,
 * no dependencies. Suitable for the "which physical channel is speaking" task
 * because the channels are already physically separated at capture — the harder
 * problem (speech vs. non-speech in the wild) is one a customer can swap in a
 * DNN VAD for via the `createVad` parameter.
 *
 * Defaults are calibrated for the dual-channel attribution use case where
 * the mic channel can pick up leakage of system audio playing through
 * speakers — in that regime, a conservative threshold (12× vs the more
 * traditional 3×) is what separates real mic speech from speaker bleed.
 * For stand-alone "is anyone talking" use cases, callers can pass a lower
 * `thresholdRatio` via the `createVad` factory.
 */
export class EnergyVad implements VadDetector {
  private readonly thresholdRatio: number;
  private readonly noiseFloorAlpha: number;
  private readonly hangoverFrames: number;
  private readonly initialNoiseFloor: number;
  private noiseFloor: number;
  private hangoverRemaining = 0;

  constructor(params: EnergyVadParams = {}) {
    this.thresholdRatio = params.thresholdRatio ?? 12;
    this.noiseFloorAlpha = params.noiseFloorAlpha ?? 0.05;
    this.hangoverFrames = params.hangoverFrames ?? 3;
    this.initialNoiseFloor = params.initialNoiseFloor ?? 1e-4;
    this.noiseFloor = this.initialNoiseFloor;
  }

  process(frame: Float32Array): VadDetectorResult {
    let sumSq = 0;
    for (let i = 0; i < frame.length; i++) {
      sumSq += frame[i] * frame[i];
    }
    const rms = frame.length > 0 ? Math.sqrt(sumSq / frame.length) : 0;

    const threshold = this.noiseFloor * this.thresholdRatio;
    let active = rms > threshold;

    if (active) {
      this.hangoverRemaining = this.hangoverFrames;
    } else if (this.hangoverRemaining > 0) {
      this.hangoverRemaining--;
      active = true;
      // While in hangover, do not update noise floor — RMS may still reflect tail energy.
    } else {
      this.noiseFloor =
        this.noiseFloor * (1 - this.noiseFloorAlpha) +
        rms * this.noiseFloorAlpha;
    }

    return { active, energy: rms, noiseFloor: this.noiseFloor };
  }

  reset(): void {
    this.noiseFloor = this.initialNoiseFloor;
    this.hangoverRemaining = 0;
  }
}
