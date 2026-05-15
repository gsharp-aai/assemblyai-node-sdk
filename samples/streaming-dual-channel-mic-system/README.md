# streaming-dual-channel-mic-system

Browser sample that streams **microphone + system audio** as a single mixed mono
stream to AssemblyAI's Streaming v3 endpoint, with per-word physical-channel
attribution (`mic` / `system`) layered on top of AAI's voice diarization
(`speaker_label`).

## Run

```bash
npm install
npm run dev
```

Then open the printed URL in Chrome, paste a streaming temporary token, and
click **Start**.

Note: the sample's `package.json` references `"assemblyai": "file:../.."` so it
builds against the local SDK source. Run `pnpm build` (or `npm run build`) once
at the SDK root before installing here.

## Getting a temporary token

API-key auth is unsupported in browsers. Mint a token from your backend:

```ts
const token = await client.streaming.createTemporaryToken({
  expires_in_seconds: 600,
})
```

## Swappable VAD

The SDK's `channelAttribution.createVad` factory is an extension point — any
class that implements `VadDetector` (`process(frame: Float32Array) → { active,
energy }` plus `reset()`) can replace the default `EnergyVad`. To plug in a
custom VAD (Silero / DNN / your own), pass a factory:

```ts
channelAttribution: {
  createVad: (channelName) => new YourCustomVadDetector(channelName),
}
```

The factory is called once per declared channel at transcriber construction
time, and the channel name (`mic` / `system` / whatever you declared in
`channels: [{ name }]`) is passed in — so factories that wrap higher-level VAD
libraries (which manage their own audio source) can map each `VadDetector`
instance to its corresponding channel.

This sample uses the default `EnergyVad` and exposes its tuning knobs via the
sliders described below.

## Resolve unknown channels

[`channelAttribution.resolveUnknownChannelsMethod`](../../src/types/streaming/index.ts)
controls how words whose per-word VAD attribution resolved to `"unknown"` are
filled in. Confident per-word VAD decisions (`"mic"` / `"system"`) are never
modified by any strategy. Default: `"window"`.

The sample's "Resolve unknown channels" dropdown switches between:

- **`window`** (default): look at the dominant non-`"unknown"` channel among
  ±2 neighboring words in the same turn. Ignores `speaker_label`, so it
  works even when AAI re-uses a label for two physically distinct voices.
  Words with no non-`"unknown"` neighbors stay `"unknown"`.
- **`speaker-history`**: accumulate per-`speaker_label` per-channel active
  VAD energy across the session. Fill `"unknown"` words with the speaker's
  dominant channel when their total evidence clears
  `speakerHistoryMinRmsEvidence` (default `0.5`) and beats runner-up by
  `speakerHistoryDominanceRatio` (default `3`). Robust when speaker labels
  are stable; does nothing when a speaker's evidence is split.
- **`none`**: disable resolution. `"unknown"` words render as-is.

Resolved words are flagged with `word.channelResolved = true`, and the sample
renders them with a trailing asterisk (e.g. `[mic*/spk A]`) so you can see
exactly when resolution fired.

### EnergyVad tuning sliders

The sample lets you tune the default
[`EnergyVad`](../../src/services/streaming/browser/energy-vad.ts) parameters
in real time:

- **Threshold ratio** (default `3`, range `1.5`–`5`, step `0.5`): the VAD
  trips when `frameRMS > noiseFloor × thresholdRatio`. Lower values are more
  sensitive (catch quieter speech, more false positives on background).
  Higher values miss quiet utterance onsets/offsets.
- **Hangover frames** (default `10` = ~200 ms, range `0`–`25`, step `5`):
  how many frames the VAD stays "active" after the last detected speech
  frame. Longer hangovers smooth attribution across brief silences within
  an utterance.

The slider values are baked into the `EnergyVad` instances created at start
time via `channelAttribution.createVad`; they cannot be changed
mid-session — Stop and Start again to apply new values.

### Speaker-change log

When `speakerLabels` is enabled and a turn's words include a transition in
the composite `(channel, speaker_label)` key vs. the previous final word,
the sample logs a line like:

    [Speaker change: mic-A → system-B]

This is the recommended pattern for transcript renderers that want to split
on speaker boundaries: compare the `(channel, speaker_label)` composite key
between consecutive words. `channel` reliably reports the physical source
(VAD-derived); `speaker_label` is AAI's acoustic diarization on the mixed
mono stream. Either change is a real boundary.

## Platform caveats

- **macOS:** `getDisplayMedia({ audio: true })` does **not** capture system
  audio by default. Install [BlackHole](https://existential.audio/blackhole/) or
  [Loopback.app](https://rogueamoeba.com/loopback/) and route system audio
  through the loopback device to make it available.
- **Windows:** sharing the whole screen via the picker exposes full system
  audio; sharing only a tab exposes just that tab's audio.
- **Speakers vs. headphones:** the energy-based VAD will misattribute when the
  mic acoustically picks up the speaker playback. Use headphones, or plug in
  a DNN-based VAD via `channelAttribution.createVad` which is more robust to
  leak.
