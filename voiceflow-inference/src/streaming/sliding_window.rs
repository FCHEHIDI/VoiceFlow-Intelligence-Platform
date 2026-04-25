//! Sliding-window segmentation for streaming diarization.
//!
//! Splits a PCM `f32` audio buffer (sampled at `sample_rate`) into overlapping
//! windows of `window_secs` seconds, advancing by `hop_secs` per window. The
//! final partial window (if any) is dropped — callers can handle it via the
//! audio buffer in `streaming::mod`.

/// Returns `(start_seconds, end_seconds, window_samples)` for each full window.
pub fn sliding_window(
    audio: &[f32],
    window_secs: f64,
    hop_secs: f64,
    sample_rate: u32,
) -> Vec<(f64, f64, Vec<f32>)> {
    assert!(window_secs > 0.0, "window_secs must be > 0");
    assert!(hop_secs > 0.0, "hop_secs must be > 0");
    assert!(sample_rate > 0, "sample_rate must be > 0");

    let sr = sample_rate as f64;
    let window_samples = (window_secs * sr).round() as usize;
    let hop_samples = (hop_secs * sr).round() as usize;
    if audio.len() < window_samples {
        return Vec::new();
    }

    let mut windows = Vec::new();
    let mut start = 0usize;
    while start + window_samples <= audio.len() {
        let end = start + window_samples;
        let start_sec = start as f64 / sr;
        let end_sec = end as f64 / sr;
        windows.push((start_sec, end_sec, audio[start..end].to_vec()));
        start += hop_samples;
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_three_windows_for_5s_at_3s_hop_1s() {
        let audio = vec![0.0_f32; 80_000];
        let w = sliding_window(&audio, 3.0, 1.0, 16_000);
        assert_eq!(w.len(), 3);
        assert!((w[0].0 - 0.0).abs() < 1e-9);
        assert!((w[0].1 - 3.0).abs() < 1e-9);
        assert_eq!(w[0].2.len(), 48_000);
        assert!((w[2].0 - 2.0).abs() < 1e-9);
        assert!((w[2].1 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn returns_empty_when_audio_shorter_than_window() {
        let audio = vec![0.0_f32; 1000];
        assert!(sliding_window(&audio, 3.0, 1.0, 16_000).is_empty());
    }

    #[test]
    fn hop_equals_window_means_non_overlapping() {
        let audio = vec![0.0_f32; 32_000];
        let w = sliding_window(&audio, 1.0, 1.0, 16_000);
        assert_eq!(w.len(), 2);
        assert!((w[1].0 - 1.0).abs() < 1e-9);
        assert!((w[1].1 - 2.0).abs() < 1e-9);
    }
}
