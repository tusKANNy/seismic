use serde::{Deserialize, Serialize};

use crate::utils::quantize;
use vectorium::{
    ComponentType, Dataset, SpaceUsage, SparseDataset, SparseVectorEncoder, SparseVectorView,
    ValueType,
};

use rustc_hash::FxHashMap;

use toolkit::BitField;
use toolkit::EliasFano;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct QuantizedSummary<C: ComponentType> {
    n_summaries: usize,
    dim: usize,
    component_ids: Option<Box<[C]>>,
    offsets: EliasFano,
    summaries_ids: BitField,
    values: Box<[u8]>,
    minimums: Box<[f32]>,
    quants: Box<[f32]>,
}

use mem_dbg::{MemSize, SizeFlags};

impl<C: ComponentType + SpaceUsage> SpaceUsage for QuantizedSummary<C> {
    fn space_usage_bytes(&self) -> usize {
        let component_ids_size = if let Some(ref component_ids) = self.component_ids {
            SpaceUsage::space_usage_bytes(component_ids)
        } else {
            0_usize
        };

        component_ids_size
            + SpaceUsage::space_usage_bytes(&self.n_summaries)
            + SpaceUsage::space_usage_bytes(&self.dim)
            + self.offsets.mem_size(SizeFlags::empty())
            + self.summaries_ids.mem_size(SizeFlags::empty())
            + SpaceUsage::space_usage_bytes(&self.values)
            + SpaceUsage::space_usage_bytes(&self.minimums)
            + SpaceUsage::space_usage_bytes(&self.quants)
    }
}

impl<C: ComponentType> QuantizedSummary<C> {
    /// Calculate space usage for sparse offset strategy (with component_ids)
    fn estimate_sparse_space(num_components: usize, max_offset: usize) -> usize {
        // Space for component_ids: number of components * size of component type C in bits
        let component_ids_space = num_components * std::mem::size_of::<C>() * 8;

        let ef_space = EliasFano::estimate_space_bits(max_offset + 1, num_components);

        component_ids_space + ef_space
    }

    /// Calculate space usage for dense offset strategy (without component_ids)
    fn estimate_dense_space(d: usize, max_offset: usize) -> usize {
        // Space for Elias-Fano offsets with full dimension d
        EliasFano::estimate_space_bits(max_offset + 1 + d, d)
    }

    /// Accumulate the contribution of one query component into the summary scores.
    ///
    /// `positions` indexes both `values` and `summaries_ids`. The arithmetic uses the algebraic
    /// operations, which let the backend contract each dequantize-and-accumulate into two fused
    /// multiply-adds; the result can differ from the strict version in the last bit.
    #[inline]
    fn accumulate(&self, positions: std::ops::Range<usize>, qv: f32, accumulator: &mut [f32]) {
        debug_assert!(positions.end <= self.values.len());
        // SAFETY: offsets are monotonic and within values bounds (validated via debug assert).
        let current_values = unsafe { self.values.get_unchecked(positions.clone()) };

        for (pos, &v) in positions.zip(current_values) {
            debug_assert!(pos < self.summaries_ids.len());
            // SAFETY: pos is within summaries_ids bounds (validated via debug assert).
            let s_id = unsafe { self.summaries_ids.get_unchecked(pos) as usize };
            debug_assert!(s_id < self.quants.len() && s_id < accumulator.len());
            let dequantized_val = unsafe {
                // SAFETY: s_id comes from summaries_ids and must be < minimums/quants len.
                (v as f32)
                    .algebraic_mul(*self.quants.get_unchecked(s_id))
                    .algebraic_add(*self.minimums.get_unchecked(s_id))
            };
            // SAFETY: s_id is within accumulator bounds (validated via debug assert).
            let slot = unsafe { accumulator.get_unchecked_mut(s_id) };
            *slot = slot.algebraic_add(dequantized_val.algebraic_mul(qv));
        }
    }

    pub fn distances<V>(&self, query: &SparseVectorView<'_, C, V>) -> Vec<f32>
    where
        V: ValueType,
    {
        let query_components = query.components();
        let query_values = query.values();

        let mut accumulator = vec![0_f32; self.n_summaries];

        if let Some(component_ids) = self.component_ids.as_ref() {
            let mut i = 0; // index for component_ids
            let mut j = 0; // index for query_components

            while i < component_ids.len() && j < query_components.len() {
                // TODO: dont do casting here once both are of type C::ComponentType
                let comp_id = component_ids[i].as_();
                let query_comp = query_components[j].as_();

                if comp_id == query_comp {
                    // Found matching component, process it
                    let qv = query_values[j];
                    let current_offset = self.offsets.select(i).unwrap();
                    let next_offset = self.offsets.select(i + 1).unwrap();

                    debug_assert!(current_offset <= next_offset);
                    self.accumulate(
                        current_offset..next_offset,
                        qv.to_f32().unwrap(),
                        &mut accumulator,
                    );

                    i += 1;
                    j += 1;
                } else if comp_id < query_comp {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        } else {
            // If there are no component_ids, we use the query_components directly
            // This is the case when we have a dense offset vector

            for (&qc, &qv) in query_components
                .iter()
                .zip(query_values)
                .take_while(|(qc, _)| (qc.as_()) < self.dim)
            {
                let current_offset = self.offsets.select(qc.as_()).unwrap() - qc.as_();
                let next_offset = self.offsets.select(qc.as_() + 1).unwrap() - (qc.as_() + 1);

                debug_assert!(current_offset <= next_offset);
                self.accumulate(
                    current_offset..next_offset,
                    qv.to_f32().unwrap(),
                    &mut accumulator,
                );

                // for i in 0..accumulator.len() {
                //     accumulator[i] = accumulator[i] * self.quants[i] + self.minimums[i] * q_vs[i];
                // }
            }
        }

        accumulator
    }

    /// Print detailed space usage breakdown for this QuantizedSummary
    pub fn print_space_usage(&self)
    where
        C: SpaceUsage,
    {
        // Calcolo la dimensione in byte di ogni campo
        let component_ids_size = if let Some(ref component_ids) = self.component_ids {
            SpaceUsage::space_usage_bytes(component_ids)
        } else {
            0_usize
        };

        let n_summaries_size = SpaceUsage::space_usage_bytes(&self.n_summaries);
        let d_size = SpaceUsage::space_usage_bytes(&self.dim);
        let offsets_size = self.offsets.mem_size(SizeFlags::empty());
        let summaries_ids_size = self.summaries_ids.mem_size(SizeFlags::empty());
        let values_size = SpaceUsage::space_usage_bytes(&self.values);
        let minimums_size = SpaceUsage::space_usage_bytes(&self.minimums);
        let quants_size = SpaceUsage::space_usage_bytes(&self.quants);

        // Calcolo il totale
        let total_size = component_ids_size
            + n_summaries_size
            + d_size
            + offsets_size
            + summaries_ids_size
            + values_size
            + minimums_size
            + quants_size;

        // Stampo il report dettagliato
        println!("\n=== QuantizedSummary Space Usage ===");
        println!(
            "Total size: {} bytes ({:.2} MB)",
            total_size,
            total_size as f64 / 1_048_576.0
        );
        println!("n_summaries: {}", self.n_summaries);
        println!("d (dimensions): {}", self.dim);

        if let Some(ref component_ids) = self.component_ids {
            println!(
                "Strategy: SPARSE (component_ids present, {} components)",
                component_ids.len()
            );
        } else {
            println!("Strategy: DENSE (no component_ids)");
        }

        println!("\n--- Component Breakdown ---");

        if component_ids_size > 0 {
            println!(
                "component_ids:   {:>12} bytes ({:>6.2}%)",
                component_ids_size,
                component_ids_size as f64 / total_size as f64 * 100.0
            );
        } else {
            println!(
                "component_ids:   {:>12} bytes ({:>6.2}%) [DENSE strategy]",
                0, 0.0
            );
        }

        println!(
            "n_summaries:     {:>12} bytes ({:>6.2}%)",
            n_summaries_size,
            n_summaries_size as f64 / total_size as f64 * 100.0
        );

        println!(
            "d:               {:>12} bytes ({:>6.2}%)",
            d_size,
            d_size as f64 / total_size as f64 * 100.0
        );

        println!(
            "offsets (EF):    {:>12} bytes ({:>6.2}%) [{} offsets]",
            offsets_size,
            offsets_size as f64 / total_size as f64 * 100.0,
            self.offsets.len()
        );

        println!(
            "summaries_ids:   {:>12} bytes ({:>6.2}%) [{} summaries {} distinct]",
            summaries_ids_size,
            summaries_ids_size as f64 / total_size as f64 * 100.0,
            self.summaries_ids.len(),
            self.n_summaries
        );

        println!(
            "values:          {:>12} bytes ({:>6.2}%) [{} values]",
            values_size,
            values_size as f64 / total_size as f64 * 100.0,
            self.values.len()
        );

        println!(
            "minimums:        {:>12} bytes ({:>6.2}%)",
            minimums_size,
            minimums_size as f64 / total_size as f64 * 100.0
        );

        println!(
            "quants:          {:>12} bytes ({:>6.2}%)",
            quants_size,
            quants_size as f64 / total_size as f64 * 100.0
        );

        println!("=====================================\n");
    }
}

impl<Q, C, V> From<SparseDataset<Q>> for QuantizedSummary<C>
where
    Q: SparseVectorEncoder<OutputComponentType = C, OutputValueType = V>,
    C: ComponentType + std::hash::Hash,
    V: ValueType,
{
    /// # Panics
    /// Panics if the number of summmaries is more than 2^16 (i.e., u16::MAX)
    fn from(dataset: SparseDataset<Q>) -> QuantizedSummary<C> {
        Self::from(&dataset)
    }
}

impl<Q, C, V> From<&SparseDataset<Q>> for QuantizedSummary<C>
where
    Q: SparseVectorEncoder<OutputComponentType = C, OutputValueType = V>,
    C: ComponentType + std::hash::Hash,
    V: ValueType,
{
    /// # Panics
    /// Panics if the number of summmaries is more than 2^16 (i.e., u16::MAX)
    fn from(dataset: &SparseDataset<Q>) -> QuantizedSummary<C> {
        assert!(
            dataset.len() <= u16::MAX as usize,
            "Number of summaries cannot be more than 2^16"
        );

        let mut inverted_pairs: FxHashMap<C, Vec<(u8, usize)>> =
            FxHashMap::with_capacity_and_hasher(1 << 16, Default::default());

        let mut minimums = Vec::with_capacity(inverted_pairs.len());
        let mut quants = Vec::with_capacity(inverted_pairs.len());

        for (doc_id, vector) in dataset.iter().enumerate() {
            let components = vector.components();
            let values = vector.values();
            let (minimum, quant, current_codes) = quantize(values);

            minimums.push(minimum);
            quants.push(quant);

            for (&c, score) in components.iter().zip(current_codes) {
                inverted_pairs.entry(c).or_default().push((score, doc_id));
            }
        }

        // Sort inverted pairs by component id
        let mut inverted_pairs: Vec<(C, Vec<(u8, usize)>)> = inverted_pairs.into_iter().collect();
        inverted_pairs.sort_by_key(|(component, _)| *component);

        // Calculate spaces for both strategies to choose the most efficient one
        let num_non_empty_components = inverted_pairs.len();
        let total_postings = inverted_pairs
            .iter()
            .map(|(_, list)| list.len())
            .sum::<usize>();

        let sparse_space = Self::estimate_sparse_space(num_non_empty_components, total_postings);
        let dense_space =
            Self::estimate_dense_space(dataset.input_dim(), dataset.input_dim() + total_postings);

        // Choose sparse strategy if it uses less space
        let use_sparse_strategy = sparse_space < dense_space;

        // println!("Sparse space: {}, Dense space: {}", sparse_space, dense_space);
        // println!("Using sparse strategy: {}", use_sparse_strategy);

        let mut summaries_ids: Vec<u64> = Vec::with_capacity(dataset.nnz());
        let mut codes = Vec::with_capacity(dataset.nnz());

        let (component_ids, offsets): (Option<Vec<C>>, Vec<usize>) = if use_sparse_strategy {
            let mut component_ids = Vec::with_capacity(num_non_empty_components);
            let mut offsets = Vec::with_capacity(num_non_empty_components + 1);
            offsets.push(0);

            for (c, ip) in inverted_pairs.iter() {
                codes.extend(ip.iter().map(|(s, _)| *s));
                summaries_ids.extend(ip.iter().map(|(_, id)| *id as u64));
                // Sparse offset strategy: store only occurring components.
                component_ids.push(*c);
                offsets.push(summaries_ids.len());
            }

            (Some(component_ids), offsets)
        } else {
            let mut offsets = Vec::with_capacity(dataset.input_dim() + 1);
            let mut current_offset = 0_usize;
            let mut next_component = 0_usize;

            for (c, ip) in inverted_pairs.iter() {
                let comp = c.as_();
                while next_component < comp {
                    offsets.push(current_offset);
                    next_component += 1;
                }

                offsets.push(current_offset);
                next_component = comp + 1;

                codes.extend(ip.iter().map(|(s, _)| *s));
                summaries_ids.extend(ip.iter().map(|(_, id)| *id as u64));
                current_offset = summaries_ids.len();
            }

            while next_component <= dataset.input_dim() {
                offsets.push(current_offset);
                next_component += 1;
            }

            // Make offsets strictly increasing for Elias-Fano.
            for (i, o) in offsets.iter_mut().enumerate() {
                *o += i;
            }

            (None, offsets)
        };

        Self {
            n_summaries: dataset.len(),
            dim: dataset.input_dim(),
            component_ids: component_ids.map(|c| c.into_boxed_slice()),
            offsets: EliasFano::from(&offsets),
            summaries_ids: BitField::from(summaries_ids),
            values: codes.into_boxed_slice(),
            minimums: minimums.into_boxed_slice(),
            quants: quants.into_boxed_slice(),
        }

        // me.print_space_usage();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use vectorium::{
        DatasetGrowable, DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable,
        PlainSparseQuantizer, SparseVectorView,
    };

    fn generate_random_sparse_dataset<C>(
        seed: u64,
        n_vecs: usize,
        dim: usize,
        min_nnz: usize,
        max_nnz: usize,
        value: f32,
    ) -> PlainSparseDataset<C, f32, DotProduct>
    where
        C: ComponentType + std::convert::TryFrom<usize>,
        <C as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
    {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let quantizer = PlainSparseQuantizer::<C, f32, DotProduct>::new(dim, dim);
        let mut dataset = PlainSparseDatasetGrowable::<C, f32, DotProduct>::new(quantizer);

        for _ in 0..n_vecs {
            let nnz = rng.random_range(min_nnz..=max_nnz);
            let mut components: Vec<usize> = (0..dim).collect();
            components.shuffle(&mut rng);
            components.truncate(nnz);
            components.sort_unstable();

            let components: Vec<C> = components
                .into_iter()
                .map(|x| C::try_from(x).unwrap())
                .collect();
            let values = vec![value; nnz];
            dataset.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }

        dataset.into()
    }

    fn generate_random_queries<C>(
        seed: u64,
        n_queries: usize,
        dim: usize,
        min_nnz: usize,
        max_nnz: usize,
    ) -> PlainSparseDatasetGrowable<C, f32, DotProduct>
    where
        C: ComponentType + std::convert::TryFrom<usize>,
        <C as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
    {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let quantizer = PlainSparseQuantizer::<C, f32, DotProduct>::new(dim, dim);
        let mut dataset = PlainSparseDatasetGrowable::<C, f32, DotProduct>::new(quantizer);

        for _ in 0..n_queries {
            let nnz = rng.random_range(min_nnz..=max_nnz);
            let mut components: Vec<usize> = (0..dim).collect();
            components.shuffle(&mut rng);
            components.truncate(nnz);
            components.sort_unstable();

            let components: Vec<C> = components
                .into_iter()
                .map(|x| C::try_from(x).unwrap())
                .collect();
            let values: Vec<f32> = (0..nnz).map(|_| rng.random::<f32>()).collect();
            dataset.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }

        dataset
    }

    fn compute_inner_product<C: ComponentType>(
        query_components: &[C],
        query_values: &[f32],
        vec_components: &[C],
        vec_values: &[f32],
    ) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut result = 0.0;

        while i < query_components.len() && j < vec_components.len() {
            let qc = query_components[i].as_();
            let vc = vec_components[j].as_();

            if qc == vc {
                result += query_values[i] * vec_values[j];
                i += 1;
                j += 1;
            } else if qc < vc {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }

    #[test]
    fn test_distances_iter() {
        let seed = 142;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate random dimensions within specified ranges
        let n_vecs = rng.random_range(50..=100);
        let dim = rng.random_range(100_000..=140_000);
        let min_nnz = 300;
        let max_nnz = 500;

        println!(
            "Test parameters: n_vecs={}, dim={}, min_nnz={}, max_nnz={}, seed={}",
            n_vecs, dim, min_nnz, max_nnz, seed
        );

        // Generate dataset with all values set to 1.0
        let dataset: PlainSparseDataset<u32, f32, DotProduct> =
            generate_random_sparse_dataset(seed, n_vecs, dim, min_nnz, max_nnz, 1.0);

        // Generate 100 random queries
        let mut queries = generate_random_queries(
            seed + 1, // Different seed for queries
            100,
            dim,
            min_nnz,
            max_nnz,
        );

        // Also add dataset itselft to the queries
        for vec in dataset.iter() {
            let components = vec.components().to_vec();
            let values = vec.values().to_vec();
            queries.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }

        // Create quantized summary
        let summary = QuantizedSummary::from(&dataset);

        // For each query, compare distances
        for (query_id, vec) in queries.iter().enumerate() {
            let query_components = vec.components();
            let query_values = vec.values();
            // Get distances using DistancesIter
            let query = SparseVectorView::new(query_components, query_values);
            let distances: Vec<f32> = summary.distances(&query);
            assert_eq!(distances.len(), dataset.len());

            // Compute distances explicitly
            let mut expected_distances = Vec::with_capacity(dataset.len());
            for vec in dataset.iter() {
                let vec_components = vec.components();
                let vec_values = vec.values();
                let distance = compute_inner_product(
                    query_components,
                    query_values,
                    vec_components,
                    vec_values,
                );
                expected_distances.push(distance);
            }

            // Compare results
            for (i, (got, expected)) in distances.iter().zip(expected_distances.iter()).enumerate()
            {
                assert!(
                    (got - expected).abs() < 1e-5,
                    "Distance mismatch for query {} vector {}: got {}, expected {} (diff: {})",
                    query_id,
                    i,
                    got,
                    expected,
                    (got - expected).abs()
                );
            }
        }
    }
}
