use crate::SeismicIndex as Index;
use crate::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};
use crate::inverted_index::Knn;

use half::f16;
use indicatif::ParallelProgressIterator;
use numpy::{PyArrayMethods, PyFixedUnicode, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::iter::{ParallelBridge, ParallelIterator};

use vectorium::{
    DotPacking8U16ScalarU8Dataset, IndexSerializer, ScalarSparseQuantizer, SparseDataset,
};

use super::{MAX_TOKEN_LEN, SeismicDataset};

type IndexQuantizer = ScalarSparseQuantizer<u16, f32, f16, vectorium::DotProduct>;
type IndexDataset = SparseDataset<IndexQuantizer>;
type DotPacking8Dataset = DotPacking8U16ScalarU8Dataset;

/// A Python wrapper around a SeismicIndex using DotPacking8 compression.
///
/// DotPacking8 stores sparse vectors with 8-bit packed components and 8-bit values scalar
/// quantized with one step per component, which reduces memory usage compared to standard
/// u16/f16 storage. It only supports u16 component types.
///
/// The build methods (`build`, `build_from_dataset`) first construct a standard index
/// and then transparently convert it to DotPacking8 format. Once built, the index supports
/// the same search operations as `SeismicIndex`.
#[pyclass(name = "SeismicIndexDotPacking8")]
pub struct SeismicIndexDotPacking8 {
    index: Index<DotPacking8Dataset>,
}

#[pymethods]
impl SeismicIndexDotPacking8 {
    pub fn get_doc_ids_in_postings(&self, list_id: usize) -> PyResult<Vec<usize>> {
        Ok(self.index.get_doc_ids_in_postings(list_id))
    }

    /// Get the dimensionality of the index (number of unique tokens).
    #[getter]
    pub fn get_dim(&self) -> PyResult<usize> {
        Ok(self.index.dim())
    }

    /// Get the number of documents in the index.
    #[getter]
    pub fn get_len(&self) -> PyResult<usize> {
        Ok(self.index.len())
    }

    /// Get the total number of non-zero entries in the index.
    #[getter]
    pub fn get_nnz(&self) -> PyResult<usize> {
        Ok(self.index.nnz())
    }

    /// Get the number of KNN neighbors per document.
    #[getter]
    pub fn knn_len(&self) -> PyResult<usize> {
        Ok(self.index.knn_len())
    }

    /// Print the estimated memory usage of the index in bytes.
    pub fn print_space_usage_byte(&self) {
        self.index.print_space_usage_byte();
    }

    /// Load a previously saved DotPacking8 index from disk.
    #[staticmethod]
    #[pyo3(signature = (index_path))]
    #[pyo3(text_signature = "(index_path)")]
    pub fn load(index_path: &str) -> PyResult<SeismicIndexDotPacking8> {
        let index: Index<DotPacking8Dataset> = Index::load_index(index_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load index: {:?}", e))
        })?;

        Ok(SeismicIndexDotPacking8 { index })
    }

    /// Save the index to disk in binary format.
    #[pyo3(signature = (path))]
    #[pyo3(text_signature = "(self, path)")]
    pub fn save(&self, path: &str) -> PyResult<()> {
        let full_path = format!("{}.index.seismic", path);
        println!("Saving ... {}", full_path);

        self.index.save_index(full_path.as_str()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to write index to '{}': {:?}",
                full_path, e
            ))
        })?;

        Ok(())
    }

    //TODO: This method is not implemented for DotPacking8.
    // The reason is that PackedSparseDataset<DotPacking8U16ScalarU8Encoder> implements
    // SeismicSearchDataset but NOT SeismicBuildDataset. If we remove the trait from
    // the method new of the KNN struct, we cannot access .components() and .values()
    // Probably there is a workaround but if a user want to use a KNN graph for a DotPacking8 index,
    // he/she can do it with the standard SeismicIndex.

    // #[pyo3(text_signature = "(self, nknn)")]
    // pub fn build_knn(&mut self, nknn: usize) {
    //     let knn = Knn::new(self.index.inverted_index(), nknn);
    //     self.index.add_knn(knn);
    // }

    /// Save the KNN graph to disk.
    #[pyo3(signature = (path))]
    #[pyo3(text_signature = "(self, path)")]
    pub fn save_knn(&self, path: &str) -> PyResult<()> {
        self.index
            .inverted_index()
            .knn()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "No KNN graph is attached to the index.",
                )
            })?
            .serialize(path)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to save KNN to '{}': {}",
                    path, e
                ))
            })?;

        Ok(())
    }

    /// Load a KNN graph from disk and attach it to the index.
    #[pyo3(signature = (knn_path, nknn=None))]
    #[pyo3(text_signature = "(self, knn_path, nknn=None)")]
    pub fn load_knn(&mut self, knn_path: &str, nknn: Option<usize>) {
        let knn = Knn::new_from_serialized(knn_path, nknn);
        self.index.add_knn(knn);
    }

    /// Build a DotPacking8-compressed index from a `.jsonl` or `.tar.gz` dataset file.
    ///
    /// Internally builds a standard u16 index and then converts it to DotPacking8 format.
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (input_path, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, max_fraction=1.5, doc_cut=15, nknn=0, knn_path=None, batched_indexing=None, input_token_to_id_map=None, load_content=true, num_threads=0))]
    #[pyo3(
        text_signature = "(input_path, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, max_fraction=1.5, doc_cut=15, nknn=0, knn_path=None, batched_indexing=None, input_token_to_id_map=None, load_content=True, num_threads=0)"
    )]
    pub fn build(
        input_path: &str,
        n_postings: usize,
        centroid_fraction: f32,
        min_cluster_size: usize,
        summary_energy: f32,
        max_fraction: f32,
        doc_cut: usize,
        nknn: usize,
        knn_path: Option<String>,
        batched_indexing: Option<usize>,
        input_token_to_id_map: Option<std::collections::HashMap<String, usize>>,
        load_content: bool,
        num_threads: usize,
    ) -> PyResult<SeismicIndexDotPacking8> {
        let _ = batched_indexing;
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let knn_config = KnnConfiguration::new(nknn, knn_path);

        let config = Configuration::default()
            .pruning_strategy(PruningStrategy::GlobalThreshold {
                n_postings,
                max_fraction,
            })
            .blocking_strategy(BlockingStrategy::RandomKmeans {
                centroid_fraction,
                min_cluster_size,
                clustering_algorithm: ClusteringAlgorithm::RandomKmeansInvertedIndexApprox {
                    doc_cut,
                },
            })
            .summarization_strategy(SummarizationStrategy::EnergyPreserving { summary_energy })
            .knn(knn_config);

        println!("\nBuilding the index (standard u16)...");
        println!("{:?}", config);

        let standard_index: Index<IndexDataset> = Index::from_file(
            &input_path.to_owned(),
            config,
            input_token_to_id_map,
            load_content,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to build index from file: {}. Error: {}",
                input_path, e
            ))
        })?;

        println!("Converting to DotPacking8...");
        let dotpacking8_index = standard_index.convert_dataset_into::<DotPacking8Dataset>();

        Ok(SeismicIndexDotPacking8 {
            index: dotpacking8_index,
        })
    }

    /// Build a DotPacking8-compressed index from an in-memory SeismicDataset.
    ///
    /// Internally builds a standard u16 index and then converts it to DotPacking8 format.
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (
        dataset,
        n_postings=3500,
        centroid_fraction=0.1,
        min_cluster_size=2,
        summary_energy=0.4,
        max_fraction=1.5,
        doc_cut=15,
        nknn=0,
        knn_path=None,
        batched_indexing=None,
        num_threads=0
    ))]
    #[pyo3(
        text_signature = "(dataset, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, max_fraction=1.5, doc_cut=15, nknn=0, knn_path=None, batched_indexing=None, num_threads=0)"
    )]
    pub fn build_from_dataset(
        dataset: SeismicDataset,
        n_postings: usize,
        centroid_fraction: f32,
        min_cluster_size: usize,
        summary_energy: f32,
        max_fraction: f32,
        doc_cut: usize,
        nknn: usize,
        knn_path: Option<String>,
        batched_indexing: Option<usize>,
        num_threads: usize,
    ) -> PyResult<SeismicIndexDotPacking8> {
        let _ = batched_indexing;
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let knn_config = KnnConfiguration::new(nknn, knn_path);

        let config = Configuration::default()
            .pruning_strategy(PruningStrategy::GlobalThreshold {
                n_postings,
                max_fraction,
            })
            .blocking_strategy(BlockingStrategy::RandomKmeans {
                centroid_fraction,
                min_cluster_size,
                clustering_algorithm: ClusteringAlgorithm::RandomKmeansInvertedIndexApprox {
                    doc_cut,
                },
            })
            .summarization_strategy(SummarizationStrategy::EnergyPreserving { summary_energy })
            .knn(knn_config);

        println!("\nBuilding the index (standard u16)...");
        println!("{:?}", config);

        let standard_index: Index<IndexDataset> = Index::from_dataset(dataset.dataset, config);

        println!("Converting to DotPacking8...");
        let dotpacking8_index = standard_index.convert_dataset_into::<DotPacking8Dataset>();

        Ok(SeismicIndexDotPacking8 {
            index: dotpacking8_index,
        })
    }

    /// Search the index with a single sparse query.
    #[pyo3(signature = (
        query_id,
        query_components,
        query_values,
        k,
        query_cut,
        heap_factor,
        n_knn=0,
        sorted=true
    ))]
    #[pyo3(
        text_signature = "(self, query_id, query_components, query_values, k, query_cut, heap_factor, n_knn=0, sorted=True)"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn search<'py>(
        &self,
        query_id: String,
        query_components: PyReadonlyArrayDyn<'py, PyFixedUnicode<MAX_TOKEN_LEN>>,
        query_values: PyReadonlyArrayDyn<'py, f32>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        sorted: bool,
    ) -> Vec<(String, f32, String)> {
        self.index
            .search(
                &query_id,
                &query_components
                    .to_vec()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>(),
                &query_values.to_vec().unwrap(),
                k,
                query_cut,
                heap_factor,
                n_knn,
                sorted,
            )
            .into_iter()
            .map(|r| r.to_tuple())
            .collect()
    }

    /// Look up document text by its string ID.
    #[pyo3(signature = (doc_id))]
    #[pyo3(text_signature = "(self, doc_id)")]
    pub fn get_doc_text(&self, doc_id: &str) -> Option<String> {
        self.index.get_doc_text(doc_id)
    }

    /// Batch search with multiple sparse queries.
    #[pyo3(signature = (
        queries_ids,
        query_components,
        query_values,
        k,
        query_cut,
        heap_factor,
        n_knn=0,
        sorted=true,
        num_threads=0
    ))]
    #[pyo3(
        text_signature = "(self, queries_ids, query_components, query_values, k, query_cut, heap_factor, n_knn=0, sorted=True, num_threads=0)"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_search<'py>(
        &self,
        queries_ids: PyReadonlyArrayDyn<'py, PyFixedUnicode<MAX_TOKEN_LEN>>,
        query_components: Bound<'py, PyList>,
        query_values: Bound<'_, PyList>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        sorted: bool,
        num_threads: usize,
    ) -> Vec<Vec<(String, f32, String)>> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let qv: Vec<Vec<f32>> = query_values
            .iter()
            .map(|i| {
                i.extract::<PyReadonlyArrayDyn<f32>>()
                    .unwrap()
                    .to_vec()
                    .unwrap()
            })
            .collect();

        let qc = query_components
            .iter()
            .map(|i| {
                let array = i
                    .extract::<PyReadonlyArrayDyn<'py, PyFixedUnicode<MAX_TOKEN_LEN>>>()
                    .unwrap();
                array
                    .to_vec()
                    .unwrap()
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let results: Vec<_> = queries_ids
            .to_vec()
            .unwrap()
            .iter()
            .zip(qc.iter())
            .zip(qv.iter())
            .par_bridge()
            .progress_count(queries_ids.len().unwrap() as u64)
            .map(|((query_id, components), values)| {
                self.index
                    .search(
                        &query_id.to_string(),
                        components,
                        values,
                        k,
                        query_cut,
                        heap_factor,
                        n_knn,
                        sorted,
                    )
                    .into_iter()
                    .map(|r| r.to_tuple())
                    .collect()
            })
            .collect();

        results
    }
}
