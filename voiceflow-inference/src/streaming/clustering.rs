//! Online speaker clustering for real-time diarization.
//!
//! Implements incremental clustering algorithms that assign speaker embeddings
//! to clusters in real-time, enabling streaming diarization with low latency.
//!
//! # Clustering Strategies
//!
//! 1. **Distance-based assignment** — pick the closest centroid within threshold.
//! 2. **Periodic re-clustering** — refine with agglomerative passes.
//! 3. **Temporal smoothing** — median filter to reduce jitter.

use ndarray::Array1;
use std::collections::HashMap;
use tracing::{debug, info};

/// Configuration for online clustering
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    /// Distance threshold for assigning to existing cluster (cosine distance)
    /// Lower = stricter (more clusters), Higher = looser (fewer clusters)
    pub distance_threshold: f32,

    /// Minimum cluster size for HDBSCAN re-clustering
    pub min_cluster_size: usize,

    /// Re-cluster every N embeddings to improve accuracy
    pub recluster_interval: usize,

    /// Window size for temporal smoothing (median filter)
    pub temporal_smoothing_window: usize,

    /// Minimum confidence score (0-1) to assign to existing cluster
    /// Below this, create new cluster
    pub min_assignment_confidence: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            distance_threshold: 0.6,
            min_cluster_size: 5,
            recluster_interval: 50,
            temporal_smoothing_window: 5,
            min_assignment_confidence: 0.7,
        }
    }
}

/// Online speaker clustering engine
pub struct OnlineClusterer {
    /// Configuration
    config: ClusterConfig,

    /// All embeddings collected so far (512-dim each)
    embeddings: Vec<Array1<f32>>,

    /// Speaker ID assignments for each embedding
    speaker_ids: Vec<usize>,

    /// Per-embedding timestamp ranges (start_seconds, end_seconds)
    timestamps: Vec<(f64, f64)>,

    /// Cluster centroids (mean embedding per speaker)
    centroids: HashMap<usize, Array1<f32>>,

    /// Count of embeddings per cluster
    cluster_counts: HashMap<usize, usize>,

    /// Next available speaker ID
    next_speaker_id: usize,
}

impl OnlineClusterer {
    /// Create a new online clusterer with default configuration
    pub fn new() -> Self {
        Self::with_config(ClusterConfig::default())
    }

    /// Create a new online clusterer with custom configuration
    pub fn with_config(config: ClusterConfig) -> Self {
        Self {
            config,
            embeddings: Vec::new(),
            speaker_ids: Vec::new(),
            timestamps: Vec::new(),
            centroids: HashMap::new(),
            cluster_counts: HashMap::new(),
            next_speaker_id: 0,
        }
    }

    /// Add an embedding with its (start, end) seconds and return the assigned speaker.
    pub fn add_embedding_at(
        &mut self,
        embedding: Array1<f32>,
        start_seconds: f64,
        end_seconds: f64,
    ) -> usize {
        let id = self.add_embedding(embedding);
        self.timestamps.push((start_seconds, end_seconds));
        id
    }

    /// Get smoothed segments as `(start_seconds, end_seconds, speaker_id)` triples.
    /// Adjacent segments belonging to the same speaker are merged.
    pub fn get_segments(&self) -> Vec<(f64, f64, usize)> {
        let smoothed = self.get_smoothed_assignments();
        let mut segments: Vec<(f64, f64, usize)> = Vec::new();
        for (idx, sid) in smoothed.iter().enumerate() {
            if let Some(&(start, end)) = self.timestamps.get(idx) {
                if let Some(last) = segments.last_mut() {
                    if last.2 == *sid && (start - last.1).abs() < 1e-6 {
                        last.1 = end;
                        continue;
                    }
                }
                segments.push((start, end, *sid));
            }
        }
        segments
    }

    /// Alias kept for parity with the agent brief.
    pub fn smooth_labels(&self) -> Vec<(f64, f64, usize)> {
        self.get_segments()
    }

    /// Add new embedding and return assigned speaker ID
    ///
    /// # Algorithm
    ///
    /// 1. Compare with existing cluster centroids
    /// 2. If close enough to existing cluster → assign to it
    /// 3. Otherwise → create new cluster
    /// 4. Update centroid for assigned cluster
    /// 5. Periodically re-cluster for refinement
    ///
    /// # Arguments
    ///
    /// * `embedding` - 512-dimensional speaker embedding vector
    ///
    /// # Returns
    ///
    /// Speaker ID (0, 1, 2, ...)
    pub fn add_embedding(&mut self, embedding: Array1<f32>) -> usize {
        // Normalize embedding to unit length for cosine distance
        let normalized_embedding = normalize(&embedding);

        // Try to assign to existing cluster
        let speaker_id = if let Some((cluster_id, distance)) =
            self.find_nearest_cluster(&normalized_embedding)
        {
            // Calculate confidence based on distance
            let confidence = 1.0 - distance;

            if confidence >= self.config.min_assignment_confidence {
                debug!(
                    "Assigned to existing cluster {} (distance: {:.3}, confidence: {:.3})",
                    cluster_id, distance, confidence
                );
                cluster_id
            } else {
                // Not confident enough, create new cluster
                let new_id = self.create_new_cluster();
                debug!(
                    "Low confidence ({:.3}), created new cluster {}",
                    confidence, new_id
                );
                new_id
            }
        } else {
            // No existing clusters, create first one
            let new_id = self.create_new_cluster();
            debug!("First cluster created: {}", new_id);
            new_id
        };

        // Store embedding and assignment
        self.embeddings.push(normalized_embedding.clone());
        self.speaker_ids.push(speaker_id);

        // Update cluster centroid
        self.update_centroid(speaker_id, &normalized_embedding);

        // Periodic re-clustering for refinement
        if self
            .embeddings
            .len()
            .is_multiple_of(self.config.recluster_interval)
        {
            info!(
                "Re-clustering {} embeddings for refinement",
                self.embeddings.len()
            );
            self.recluster();
        }

        speaker_id
    }

    /// Find nearest cluster centroid
    ///
    /// Returns (cluster_id, distance) if any clusters exist
    fn find_nearest_cluster(&self, embedding: &Array1<f32>) -> Option<(usize, f32)> {
        if self.centroids.is_empty() {
            return None;
        }

        let (best_cluster, min_distance) = self
            .centroids
            .iter()
            .map(|(&cluster_id, centroid)| {
                let distance = cosine_distance(embedding, centroid);
                (cluster_id, distance)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;

        // Only return if within threshold
        if min_distance <= self.config.distance_threshold {
            Some((best_cluster, min_distance))
        } else {
            None
        }
    }

    /// Create a new cluster and return its ID
    fn create_new_cluster(&mut self) -> usize {
        let cluster_id = self.next_speaker_id;
        self.next_speaker_id += 1;
        self.cluster_counts.insert(cluster_id, 0);
        cluster_id
    }

    /// Update cluster centroid with new embedding
    ///
    /// Uses incremental mean formula:
    /// new_mean = old_mean + (new_value - old_mean) / n
    fn update_centroid(&mut self, cluster_id: usize, embedding: &Array1<f32>) {
        let count = self.cluster_counts.entry(cluster_id).or_insert(0);
        *count += 1;

        if *count == 1 {
            // First embedding in cluster
            self.centroids.insert(cluster_id, embedding.clone());
        } else {
            // Update centroid incrementally
            let centroid = self.centroids.get_mut(&cluster_id).unwrap();
            let n = *count as f32;

            for i in 0..embedding.len() {
                centroid[i] += (embedding[i] - centroid[i]) / n;
            }

            // Re-normalize
            let norm = centroid.mapv(|x| x * x).sum().sqrt();
            if norm > 0.0 {
                centroid.mapv_inplace(|x| x / norm);
            }
        }
    }

    /// Re-cluster all embeddings for improved accuracy
    ///
    /// This is expensive but improves clustering quality.
    /// Only called periodically (every recluster_interval embeddings).
    fn recluster(&mut self) {
        if self.embeddings.len() < self.config.min_cluster_size * 2 {
            debug!("Too few embeddings for re-clustering, skipping");
            return;
        }

        // Use agglomerative clustering (simpler than HDBSCAN, no external deps)
        let new_assignments = self.agglomerative_clustering();

        // Update speaker IDs and rebuild centroids
        self.speaker_ids = new_assignments;
        self.rebuild_centroids();

        info!(
            "Re-clustering complete: {} embeddings → {} clusters",
            self.embeddings.len(),
            self.centroids.len()
        );
    }

    /// Simple agglomerative clustering implementation
    ///
    /// Algorithm:
    /// 1. Start with each embedding as its own cluster
    /// 2. Iteratively merge closest clusters
    /// 3. Stop when all clusters are sufficiently far apart
    fn agglomerative_clustering(&self) -> Vec<usize> {
        let n = self.embeddings.len();
        let mut assignments = (0..n).collect::<Vec<_>>();
        let mut active_clusters: Vec<usize> = (0..n).collect();

        loop {
            // Find closest pair of clusters
            let (i, j, min_dist) = self.find_closest_cluster_pair(&active_clusters);

            // Stop if closest pair is too far apart
            if min_dist > self.config.distance_threshold {
                break;
            }

            // Merge cluster j into cluster i
            let cluster_i = active_clusters[i];
            let cluster_j = active_clusters[j];

            for assignment in &mut assignments {
                if *assignment == cluster_j {
                    *assignment = cluster_i;
                }
            }

            // Remove cluster j from active list
            active_clusters.remove(j);

            // Stop if we have few enough clusters
            if active_clusters.len() <= 2 {
                break;
            }
        }

        // Renumber clusters to be contiguous (0, 1, 2, ...)
        let mut cluster_map = HashMap::new();
        let mut next_id = 0;

        for assignment in &mut assignments {
            let new_id = cluster_map.entry(*assignment).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *assignment = *new_id;
        }

        assignments
    }

    /// Find the two closest clusters
    ///
    /// Returns (index_i, index_j, distance) where index_i < index_j
    fn find_closest_cluster_pair(&self, active_clusters: &[usize]) -> (usize, usize, f32) {
        let mut min_dist = f32::INFINITY;
        let mut best_i = 0;
        let mut best_j = 1;

        for i in 0..active_clusters.len() {
            for j in (i + 1)..active_clusters.len() {
                let cluster_i = active_clusters[i];
                let cluster_j = active_clusters[j];

                // Average distance between all pairs in two clusters
                let mut total_dist = 0.0;
                let mut count = 0;

                for (idx, assignment) in self.speaker_ids.iter().enumerate() {
                    if *assignment == cluster_i {
                        for (idx2, assignment2) in self.speaker_ids.iter().enumerate() {
                            if *assignment2 == cluster_j {
                                total_dist +=
                                    cosine_distance(&self.embeddings[idx], &self.embeddings[idx2]);
                                count += 1;
                            }
                        }
                    }
                }

                let avg_dist = if count > 0 {
                    total_dist / count as f32
                } else {
                    f32::INFINITY
                };

                if avg_dist < min_dist {
                    min_dist = avg_dist;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        (best_i, best_j, min_dist)
    }

    /// Rebuild centroids from current assignments
    fn rebuild_centroids(&mut self) {
        self.centroids.clear();
        self.cluster_counts.clear();

        // Group embeddings by cluster
        let mut cluster_embeddings: HashMap<usize, Vec<Array1<f32>>> = HashMap::new();

        for (embedding, &speaker_id) in self.embeddings.iter().zip(&self.speaker_ids) {
            cluster_embeddings
                .entry(speaker_id)
                .or_default()
                .push(embedding.clone());
        }

        // Compute mean for each cluster
        for (cluster_id, embeddings) in cluster_embeddings {
            let n = embeddings.len();
            let mut centroid = Array1::zeros(512);

            for emb in &embeddings {
                centroid = &centroid + emb;
            }

            centroid /= n as f32;

            // Normalize
            let norm = centroid.mapv(|x| x * x).sum().sqrt();
            if norm > 0.0 {
                centroid /= norm;
            }

            self.centroids.insert(cluster_id, centroid);
            self.cluster_counts.insert(cluster_id, n);
        }
    }

    /// Get current speaker assignments with temporal smoothing
    ///
    /// Applies median filter to reduce jitter in speaker transitions
    pub fn get_smoothed_assignments(&self) -> Vec<usize> {
        if self.speaker_ids.len() < self.config.temporal_smoothing_window {
            return self.speaker_ids.clone();
        }

        let window = self.config.temporal_smoothing_window;
        let mut smoothed = Vec::with_capacity(self.speaker_ids.len());

        for i in 0..self.speaker_ids.len() {
            // Get window around current position
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(self.speaker_ids.len());

            let window_values = &self.speaker_ids[start..end];

            // Compute mode (most frequent value)
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for &val in window_values {
                *counts.entry(val).or_insert(0) += 1;
            }

            let mode = counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&val, _)| val)
                .unwrap_or(self.speaker_ids[i]);

            smoothed.push(mode);
        }

        smoothed
    }

    /// Get number of detected speakers
    pub fn num_speakers(&self) -> usize {
        self.centroids.len()
    }

    /// Get total number of embeddings processed
    pub fn num_embeddings(&self) -> usize {
        self.embeddings.len()
    }

    /// Reset the clusterer (clear all state)
    pub fn reset(&mut self) {
        self.embeddings.clear();
        self.speaker_ids.clear();
        self.timestamps.clear();
        self.centroids.clear();
        self.cluster_counts.clear();
        self.next_speaker_id = 0;
    }
}

impl Default for OnlineClusterer {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize vector to unit length
fn normalize(vec: &Array1<f32>) -> Array1<f32> {
    let norm = vec.mapv(|x| x * x).sum().sqrt();
    if norm > 0.0 {
        vec / norm
    } else {
        vec.clone()
    }
}

/// Compute cosine distance between two vectors
///
/// cosine_distance = 1 - cosine_similarity
/// = 1 - (a · b) / (||a|| * ||b||)
///
/// For unit vectors: cosine_distance = 1 - (a · b)
///
/// Range: [0, 2] where 0 = identical, 2 = opposite
fn cosine_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    // For normalized vectors, cosine similarity = dot product
    let cosine_similarity = dot_product;

    // Convert to distance: 0 = same, 2 = opposite
    1.0 - cosine_similarity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let vec = Array1::from_vec(vec![3.0, 4.0]);
        let normalized = normalize(&vec);

        // Should be unit length
        let norm = normalized.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        // Identical vectors
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0]);
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);

        // Perpendicular vectors
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);

        // Opposite vectors
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0]);
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_basic_clustering() {
        let mut clusterer = OnlineClusterer::new();

        // Add two clearly distinct embeddings
        let emb1 = Array1::from_vec(vec![1.0; 512]);
        let emb2 = Array1::from_vec(vec![-1.0; 512]);

        let id1 = clusterer.add_embedding(emb1.clone());
        let id2 = clusterer.add_embedding(emb2);

        // Should create two separate clusters
        assert_ne!(id1, id2);
        assert_eq!(clusterer.num_speakers(), 2);

        // Add similar to first
        let emb3 = Array1::from_vec(vec![0.9; 512]);
        let id3 = clusterer.add_embedding(emb3);

        // Should assign to first cluster
        assert_eq!(id3, id1);
        assert_eq!(clusterer.num_speakers(), 2);
    }
}
