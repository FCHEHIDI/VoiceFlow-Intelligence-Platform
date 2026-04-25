//! Integration tests covered by the broader pipeline (sliding window, clustering, audio buffer).
//! Pipeline-level pieces (sliding window, online clustering) live inside the crate and are
//! exercised by their own `#[cfg(test)]` modules; here we cover the public surface reachable
//! from external integration tests.

#[cfg(test)]
mod tests {
    use voiceflow_inference::streaming::AudioBuffer;

    #[test]
    fn audio_buffer_fills_to_capacity() {
        let mut buffer = AudioBuffer::new(1600);
        buffer.push(&vec![0.5_f32; 800]);
        assert_eq!(buffer.len(), 800);
        assert!(!buffer.is_full());

        buffer.push(&vec![0.5_f32; 800]);
        assert_eq!(buffer.len(), 1600);
        assert!(buffer.is_full());
    }

    #[test]
    fn audio_buffer_is_fifo_when_overflowing() {
        let mut buffer = AudioBuffer::new(4);
        buffer.push(&[1.0, 2.0, 3.0, 4.0]);
        assert!(buffer.is_full());
        buffer.push(&[5.0, 6.0]);
        assert_eq!(buffer.get_chunk(), &[3.0, 4.0, 5.0, 6.0]);
    }
}

// ---------------------------------------------------------------------------
// Sliding-window integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod sliding_window_tests {
    use voiceflow_inference::streaming::sliding_window::sliding_window;

    const SAMPLE_RATE: u32 = 16_000;

    fn fake_audio(seconds: f64) -> Vec<f32> {
        vec![0.0_f32; (seconds * SAMPLE_RATE as f64) as usize]
    }

    /// 10 s of audio with a 3 s window and 1 s hop -> 8 full windows starting
    /// at 0, 1, ..., 7 seconds. The trailing partial window (8..10 s) is
    /// dropped per the implementation in `streaming/sliding_window.rs`.
    #[test]
    fn ten_seconds_window3_hop1_yields_eight_windows() {
        let audio = fake_audio(10.0);
        let windows = sliding_window(&audio, 3.0, 1.0, SAMPLE_RATE);

        assert_eq!(windows.len(), 8, "expected 8 full 3 s windows in 10 s audio");

        for (i, (start, end, samples)) in windows.iter().enumerate() {
            let expected_start = i as f64;
            let expected_end = expected_start + 3.0;
            assert!(
                (*start - expected_start).abs() < 1e-9,
                "window {} start {} != {}",
                i,
                start,
                expected_start
            );
            assert!(
                (*end - expected_end).abs() < 1e-9,
                "window {} end {} != {}",
                i,
                end,
                expected_end
            );
            assert_eq!(
                samples.len(),
                3 * SAMPLE_RATE as usize,
                "window {} has wrong sample count",
                i
            );
        }
    }

    /// 2 s of audio with a 3 s window -> the audio is shorter than one
    /// window, so the function returns an empty Vec (no partial window).
    #[test]
    fn two_seconds_with_three_second_window_yields_no_windows() {
        let audio = fake_audio(2.0);
        let windows = sliding_window(&audio, 3.0, 1.0, SAMPLE_RATE);
        assert!(
            windows.is_empty(),
            "expected 0 windows, got {}",
            windows.len()
        );
    }

    #[test]
    fn exactly_three_seconds_yields_a_single_window() {
        let audio = fake_audio(3.0);
        let windows = sliding_window(&audio, 3.0, 1.0, SAMPLE_RATE);
        assert_eq!(windows.len(), 1);
        assert!((windows[0].0 - 0.0).abs() < 1e-9);
        assert!((windows[0].1 - 3.0).abs() < 1e-9);
        assert_eq!(windows[0].2.len(), 3 * SAMPLE_RATE as usize);
    }
}

// ---------------------------------------------------------------------------
// Online clusterer integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod clustering_tests {
    use ndarray::Array1;
    use voiceflow_inference::streaming::clustering::OnlineClusterer;

    /// Build a 512-dim unit-vector embedding with `1.0` on `axis` and 0 elsewhere.
    fn unit_axis(axis: usize) -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(512);
        v[axis] = 1.0;
        v
    }

    #[test]
    fn same_embedding_added_twice_is_same_speaker() {
        let mut clusterer = OnlineClusterer::default();
        let emb = unit_axis(0);

        let id1 = clusterer.add_embedding_at(emb.clone(), 0.0, 1.0);
        let id2 = clusterer.add_embedding_at(emb.clone(), 1.0, 2.0);

        assert_eq!(id1, id2, "identical embeddings should map to the same speaker");
        assert_eq!(clusterer.num_speakers(), 1);

        let segments = clusterer.get_segments();
        // After temporal smoothing & merge of adjacent same-speaker segments,
        // we expect a single contiguous segment 0.0..2.0 for that speaker.
        assert!(!segments.is_empty(), "expected at least one segment");
        let speakers: std::collections::HashSet<usize> =
            segments.iter().map(|(_, _, s)| *s).collect();
        assert_eq!(
            speakers.len(),
            1,
            "all segments should belong to one speaker"
        );
    }

    #[test]
    fn orthogonal_embeddings_produce_distinct_speakers() {
        let mut clusterer = OnlineClusterer::default();
        let e1 = unit_axis(0); // [1, 0, 0, ..., 0]
        let e2 = unit_axis(1); // [0, 1, 0, ..., 0]

        let id1 = clusterer.add_embedding_at(e1, 0.0, 1.0);
        let id2 = clusterer.add_embedding_at(e2, 1.0, 2.0);

        assert_ne!(
            id1, id2,
            "orthogonal embeddings must be assigned different speakers"
        );
        assert_eq!(clusterer.num_speakers(), 2);
    }

    #[test]
    fn three_alternating_embeddings_produce_two_speakers() {
        let mut clusterer = OnlineClusterer::default();
        let e_a = unit_axis(0);
        let e_b = unit_axis(1);

        let id_a1 = clusterer.add_embedding_at(e_a.clone(), 0.0, 1.0);
        let id_b = clusterer.add_embedding_at(e_b, 1.0, 2.0);
        let id_a2 = clusterer.add_embedding_at(e_a, 2.0, 3.0);

        assert_ne!(id_a1, id_b);
        assert_eq!(id_a1, id_a2);
        assert_eq!(clusterer.num_speakers(), 2);
    }
}
