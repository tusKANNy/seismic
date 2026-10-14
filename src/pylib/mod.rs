use crate::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};
use crate::inverted_index::Knn;

use crate::SeismicIndex as Index;
use half::f16;

use indicatif::ParallelProgressIterator;
use numpy::{PyArrayMethods, PyFixedUnicode, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::iter::{ParallelBridge, ParallelIterator};

use std::collections::HashMap;

use crate::InvertedIndexBase;
use vectorium::{
    ComponentType, Dataset, Distance, DotProduct, IndexSerializer, PlainSparseDataset,
    ScalarSparseQuantizer, SparseDataset, SparseVectorView,
};

const MAX_TOKEN_LEN: usize = 30;
const SEISMIC_STRING: &str = "U30";

type IndexQuantizer<C> = ScalarSparseQuantizer<C, f32, f16, DotProduct>;
type IndexDataset<C> = SparseDataset<IndexQuantizer<C>>;

fn quantize_f32_dataset_to_f16<C>(
    dataset: PlainSparseDataset<C, f32, DotProduct>,
) -> IndexDataset<C>
where
    C: ComponentType,
    IndexDataset<C>: vectorium::dataset::ConvertFrom<PlainSparseDataset<C, f32, DotProduct>>,
{
    use vectorium::dataset::ConvertFrom;
    IndexDataset::<C>::convert_from(dataset)
}

#[pyfunction]
pub fn get_seismic_string() -> &'static str {
    SEISMIC_STRING
}

macro_rules! impl_seismic_index {
    ($rust_name:ident, $py_name:literal, $Key:ty, $dataset_type:ident) => {
        /// A Python wrapper around the SeismicIndex data structure.
        ///
        /// This class provides methods to build, query, and manage an inverted index
        /// over a sparse dataset. Typically, it is constructed from a file or an in-memory
        /// dataset using `build` or `build_from_dataset`. See these methods for further details.
        #[pyclass(name= $py_name)]
        pub struct $rust_name {
            index: Index<IndexDataset<$Key>>,
        }

        #[pymethods]
        impl $rust_name {

            pub fn get_doc_ids_in_postings(&self, list_id: usize) -> PyResult<Vec<usize>> {
                Ok(self.index.inverted_index().get_doc_ids_in_postings(list_id))
            }

            /// Get the dimensionality of the index.
            ///
            /// This method returns the total number of unique tokens
            /// present in the dataset used to build the index.
            ///
            /// Returns:
            ///     int: The number of dimensions.
            ///
            /// Example:
            ///     >>> index.dim
            ///     128
            #[getter]
            pub fn get_dim(&self) -> PyResult<usize> {
                Ok(self.index.dim())
            }


            /// Get the number of documents in the index.
            ///
            /// This method returns the total number of documents (or vectors)
            /// that were indexed.
            ///
            /// Returns:
            ///     int: The number of documents in the index.
            ///
            /// Example:
            ///     >>> index.len
            ///     10000
            #[getter]
            pub fn get_len(&self) -> PyResult<usize> {
                Ok(self.index.len())
            }

            /// Get the number of non-zero entries (NNZ) in the index.
            ///
            /// This method returns the total number of token-value pairs across all documents,
            /// i.e., the total number of sparse components stored in the index.
            ///
            /// Returns:
            ///     int: The number of non-zero entries.
            ///
            /// Example:
            ///     >>> index.nnz
            ///     532000
            #[getter]
            pub fn get_nnz(&self) -> PyResult<usize> {
                Ok(self.index.nnz())
            }

            /// Get the number of neighbors per each document stored in the index (that would be the "k" for the k-NN graph).
            ///
            /// This method returns the number of precomputed k-nearest neighbor available per document.
            ///
            /// Returns:
            ///     int: Number of precomputed neighbors (K) per document.
            ///
            /// Example:
            ///     >>> index.knn_len
            ///     10
            #[getter]
            pub fn knn_len(&self) -> PyResult<usize> {
                Ok(self.index.knn_len())
            }

            /// Print the estimated memory usage of the index in bytes.
            ///
            /// This method returns no value, but logs to the console the memory
            /// footprint of the inverted index and associated data structures.
            ///
            /// Example:
            ///     >>> index.print_space_usage_byte()
            ///     Total space usage: 12.3 MB
            pub fn print_space_usage_byte(&self) {
                self.index.print_space_usage_byte();
            }

            /// Get the sparse vector representation of a document by its ID.
            ///
            /// This method returns the list of token IDs and their associated values
            /// for the specified document index.
            ///
            /// Args:
            ///     id (int): The document ID.
            ///
            /// Returns:
            ///        tuple[list[int], list[float]]: A pair of lists containing token IDs and corresponding values.
            ///
            /// Example:
            ///     >>> tokens, values = index.get(42)
            ///     >>> print(tokens)
            ///     [3, 7, 19]
            ///
            #[pyo3(signature = (id))]
            #[pyo3(text_signature = "(self, id)")]
            pub fn get(&self, id: usize) -> PyResult<(Vec<$Key>, Vec<f32>)> {
                let entry = self.index.dataset().get(id as u64);
                Ok((
                    entry.components().to_vec(),
                    entry.values().iter().map(|v| v.to_f32()).collect(),
                ))
            }

            /// Load a previously saved SeismicIndex from disk.
            ///
            /// This method returns a deserialized `SeismicIndex` from the given file path.
            /// The file should have been created using the `.save()` method.
            ///
            /// Args:
            ///     index_path (str): Path to the `.index.seismic` file.
            ///
            /// Returns:
            ///     SeismicIndex: The loaded index instance.
            ///
            /// Raises:
            ///     IOError: If the file cannot be found or deserialized.
            ///
            /// Example:
            ///     >>> index = SeismicIndex.load("my_index.index.seismic")
            #[staticmethod]
            #[pyo3(signature = (index_path))]
            #[pyo3(text_signature = "(index_path)")]
            pub fn load(index_path: &str) -> PyResult<$rust_name> {
                let index: Index<IndexDataset<$Key>> = Index::load_index(index_path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load index: {:?}", e)))?;

                Ok($rust_name { index })
            }

            /// Save the current SeismicIndex to disk in binary format.
            ///
            /// This method returns nothing, but writes a `.index.seismic` file to the specified path.
            /// The file can later be loaded using the `SeismicIndex.load()` method.
            ///
            /// Args:
            ///     path (str): Path (without extension) where the index will be saved.
            ///
            /// Raises:
            ///     IOError: If serialization or writing to the file fails.
            ///
            /// Example:
            ///     >>> index.save("my_index")
            ///     # Creates a file named "my_index.index.seismic"
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

            /// Build and attach a KNN (k-nearest neighbor) graph to the index.
            ///
            /// This method returns nothing, but internally computes the nearest neighbors
            /// for each document in the index and stores the result. The nearest neighbors are
            /// identified by using the seismic index itself, so they are approximated.
            ///
            /// Args:
            ///     nknn (int): The number of nearest neighbors to compute for each document.
            ///
            /// Example:
            ///     >>> index.build_knn(10)
            #[pyo3(signature = (nknn))]
            #[pyo3(text_signature = "(self, nknn)")]
            pub fn build_knn(&mut self, nknn: usize) {
                let knn = Knn::new(self.index.inverted_index(), nknn);
                self.index.add_knn(knn);
            }

            /// Save the precomputed KNN graph to disk.
            ///
            /// This method returns nothing, but writes the KNN data to the given file path.
            /// The file can later be reloaded using `load_knn()`.
            ///
            /// Args:
            ///     path (str): Path where the KNN file will be saved.
            ///
            /// Raises:
            ///     IOError: If serialization or writing fails.
            ///
            /// Example:
            ///     >>> index.save_knn("my_index.knn")
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

            /// Load a precomputed KNN graph from a file and attach it to the index.
            ///
            /// This method reads a serialized KNN structure and attaches it to the existing index.
            ///
            /// Args:
            ///     knn_path (str): Path to the KNN file.
            ///     nknn (int, optional): Number of neighbors to read (You may have x neighbors per document in your k-NN, but add only y<x in your file).
            ///
            /// Example:
            ///     >>> index.load_knn("my_index.knn")
            #[pyo3(signature = (knn_path, nknn=None))]
            #[pyo3(text_signature = "(self, knn_path, nknn=None)")]
            pub fn load_knn(&mut self, knn_path: &str, nknn: Option<usize>) {
                let knn = Knn::new_from_serialized(knn_path, nknn);
                self.index.add_knn(knn);
            }

             /*
        Order of attributes matters:
        Rust processes attributes sequentially, and attributes like #[pyo3(...)] and #[staticmethod] interact with each other. By placing:
        #[allow(clippy::too_many_arguments)] above #[staticmethod], it ensures Clippy's lint suppression happens first without interfering
        with Pyo3's attribute processing. When #[staticmethod] comes first, it can cause unexpected behavior if the following attributes
        aren't interpreted correctly. */

        /// Build a new SeismicIndex from a `.jsonl` or `.tar.gz` dataset file.
        ///
        /// This method processes the input dataset and builds a Seimic inverted index.
        ///
        /// Parameters:
        ///     input_path (str): Path to the dataset file.
        ///     n_postings (int): Number of average postings per token (default 3500).
        ///     centroid_fraction (float): Fraction of documents in each inverted list used to form centroids (on average).
        ///     min_cluster_size (int): Minimum number of documents per cluster.
        ///     summary_energy (float): Target energy retention for summarization.
        ///     max_fraction (float): Maximum number of document to keep in each inverted list computed as n_postings * max_fraction.
        ///     doc_cut (int): Thereshold for document approximation at building time.
        ///     nknn (int, optional): Number of nearest neighbors to compute.
        ///     knn_path (str, optional): Path to precomputed KNN data.
        ///     batched_indexing (int, optional): Batch size for indexing. Reducing this values reduces peak memory usage.
        ///     input_token_to_id_map (dict, optional): Predefined token-to-ID mapping. If None, the index will build its own mapping.
        ///     load_content (bool, optional): Whether to load document content (default: True).
        ///     num_threads (int, optional): Number of threads to use (default: use Rayon default).
        ///
        /// Returns:
        ///     SeismicIndex: A fully constructed inverted index.
        ///
        /// Raises:
        ///     IOError: If the file cannot be read or parsed.
        ///
        /// Example:
        ///     >>> index = seismic.SeismicIndex.build("data.tar.gz", n_postings=4000)
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
            input_token_to_id_map: Option<HashMap<String, usize>>,
            load_content: bool,
            num_threads: usize,
        ) -> PyResult<$rust_name> {
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

            println!("\nBuilding the index...");
            println!("{:?}", config);

            let index = Index::from_file(&input_path.to_owned(), config, input_token_to_id_map, load_content)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to build index from file: {}. File may not exist or be corrupted. Error: {}",
                        input_path,
                        e
                    ))
                })?;

            Ok($rust_name { index })
        }
        /// Build a SeismicIndex from an in-memory SeismicDataset.
        ///
        /// This method returns a new index by processing the provided dataset with the given configuration.
        /// It supports pruning, clustering, summarization, and optional KNN graph construction.
        ///
        /// Args:
        ///     dataset (SeismicDataset): The dataset to index.
        ///     n_postings (int, optional): Max number of postings per token (default: 3500).
        ///     centroid_fraction (float, optional): Fraction of documents in each inverted list used to form centroids (default: 0.1).
        ///     min_cluster_size (int, optional): Minimum number of documents per cluster (default: 2).
        ///     summary_energy (float, optional): Target energy retention for summarization (default: 0.4).
        ///     max_fraction (float): Maximum number of document to keep in each inverted list computed as n_postings * max_fraction.
        ///     doc_cut (int): Thereshold for document approximation at building time.
        ///     nknn (int, optional): Number of nearest neighbors to compute (default: 0).
        ///     knn_path (str, optional): Path to a precomputed KNN file (optional).
        ///     batched_indexing (int, optional): Indexing batch size to reduce memory usage (optional).
        ///     num_threads (int, optional): Number of threads to use (default: Rayon default).
        ///
        /// Returns:
        ///     SeismicIndex: The constructed inverted index.
        ///
        /// Example:
        ///     >>> index = SeismicIndex.build_from_dataset(dataset, n_postings=5000)
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
            dataset: $dataset_type,
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
        ) -> PyResult<$rust_name> {
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

            println!("\nBuilding the index...");
            println!("{:?}", config);

            let index = Index::from_dataset(dataset.dataset, config);

            Ok($rust_name { index })
        }

        /// Perform a nearest neighbor search over the index using a single sparse query.
        ///
        /// This method returns the top-k most similar documents to the input query based on
        /// inner (dot) product.
        ///
        /// Args:
        ///     query_id (str): Identifier for the query (used for result annotation).
        ///     query_components (ndarray[str]): List of tokens (components) in the query.
        ///     query_values (ndarray[float32]): Corresponding values for each token.
        ///     k (int): Number of results to return.
        ///     query_cut (int): Maximum number of tokens considered from the query.
        ///     heap_factor (float): Heap factor used during search.
        ///     n_knn (int, optional): Number of KNN neighbors to scan for result refinement. Requires the index to have a k-NN graph (default: 0).
        ///     sorted (bool, optional): Whether to scan the summaries in each posting lists starting from the most similar (default: True).
        ///
        /// Returns:
        ///     list[tuple[str, float, str]]: A list of (query_id, distance, document_id) tuples.
        ///
        /// Example:
        ///     >>> index.search("q1", np.array(["token1", "token2"], dtype=seismic.get_seismic_string()), np.array([0.5, 0.3], dtype=np.float32), k=5, query_cut=10, heap_factor=0.8)
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
            self.index.search(
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

        /// Look up the stored text content for a document by its string document ID.
        ///
        /// Args:
        ///     doc_id (str): The document ID as returned by `search()`.
        ///
        /// Returns:
        ///     str | None: The document text if content was loaded at build time, otherwise None.
        ///
        /// Example:
        ///     >>> text = index.get_doc_text("doc42")
        #[pyo3(signature = (doc_id))]
        #[pyo3(text_signature = "(self, doc_id)")]
        pub fn get_doc_text(&self, doc_id: &str) -> Option<String> {
            self.index.get_doc_text(doc_id)
        }

        /// Perform batched nearest neighbor search using multiple sparse query vectors.
    ///
    /// This method returns the top-k most similar documents according to the inner product, for each query.
    /// It supports parallel search, see `num_threads` for more details.
    ///
    /// Args:
    ///     queries_ids (ndarray[str]): Array of query ids, one per query.
    ///     query_components (list[ndarray[str]]): List of arrays, each containing the tokens of a query.
    ///     query_values (list[ndarray[float32]]): List of arrays, each containing values corresponding to tokens.
    ///     k (int): Number of results to return per query.
    ///     query_cut (int): Maximum number of tokens considered per query.
    ///     heap_factor (float): Heap factor used during search.
    ///     n_knn (int, optional): Number of KNN neighbors to scan for refinement. Requires the index to have a k-NN graph (default: 0).
    ///     sorted (bool, optional): Whether to scan the summaries in each posting lists starting from the most similar one (default: True).
    ///     num_threads (int, optional): Number of threads to use for batch execution (default: 0 = Rayon default).
    ///
    /// Returns:
    ///     list[list[tuple[str, float, str]]]: A list of result lists, one per query. Each result is a (query_id, distance, document_id) tuple.
    ///
    /// Example:
    ///     >>> results = index.batch_search(query_ids, query_components, query_values, k=10, query_cut=20, heap_factor=0.8)
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
                self.index.search(
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
    };
}

macro_rules! impl_seismic_index_raw {
      ($rust_name:ident, $py_name:literal, $Key:ty) => {
        /// A Python wrapper around a raw Seismic inverted index.
        ///
        /// This class provides a lightweight interface to a compressed sparse index
        /// without requiring the full document/token mapping. It is typically used for
        /// performing efficient search over preprocessed binary datasets.
        ///
        /// Unlike `SeismicIndex`, this version does not include document metadata or
        /// token-to-string resolution. It's ideal for inference and scenarios where
        /// document IDs and token strings are not needed.
        ///
        #[pyclass(name= $py_name)]
        pub struct $rust_name {
            inverted_index: InvertedIndexBase<IndexDataset<$Key>>,
        }


        #[pymethods]
        impl $rust_name {

            pub fn get_doc_ids_in_postings(&self, list_id: usize) -> PyResult<Vec<usize>> {
                Ok(self.inverted_index.get_doc_ids_in_postings(list_id))
            }

            /// Get the dimensionality of the raw index.
            ///
            /// This method returns the total number of unique tokens (features)
            /// present in the dataset used to build the raw index.
            ///
            /// Returns:
            ///     int: The number of dimensions.
            ///
            /// Example:
            ///     >>> index.dim
            ///     128
            #[getter]
            pub fn get_dim(&self) -> PyResult<usize> {
                Ok(self.inverted_index.dim())
            }

            /// Get the number of documents in the raw index.
            ///
            /// This method returns the total number of vectors (documents)
            /// stored in the inverted index.
            ///
            /// Returns:
            ///     int: The number of indexed documents.
            ///
            /// Example:
            ///     >>> index.len
            ///     10000
            #[getter]
            pub fn get_len(&self) -> PyResult<usize> {
                Ok(self.inverted_index.len())
            }



            /// Get the number of non-zero entries (NNZ) in the raw index.
            ///
            /// This method returns the total number of token-value pairs across all
            /// documents stored in the inverted index.
            ///
            /// Returns:
            ///     int: The total number of non-zero components.
            ///
            /// Example:
            ///     >>> index.nnz
            ///     531287
            #[getter]
            pub fn get_nnz(&self) -> PyResult<usize> {
                Ok(self.inverted_index.nnz())
            }

            /// Get the number of precomputed KNN neighbors per document.
            ///
            /// This method returns the number of nearest neighbors stored
            /// for each document in the KNN graph attached to the raw index.
            ///
            /// Returns:
            ///     int: The number of neighbors per document.
            ///
            /// Example:
            ///     >>> index.knn_len
            ///     10
            #[getter]
            pub fn knn_len(&self) -> PyResult<usize> {
                Ok(self.inverted_index.knn_len())
            }

            /// Check whether the raw index contains any documents.
            ///
            /// This method returns `True` if the index is empty (i.e., contains no vectors),
            /// and `False` otherwise.
            ///
            /// Returns:
            ///     bool: Whether the index is empty.
            ///
            /// Example:
            ///     >>> index.is_empty
            ///     False
            #[getter]
            pub fn get_is_empty(&self) -> PyResult<bool> {
                Ok(self.inverted_index.is_empty())
            }

            /// Print the memory usage of the raw index in bytes.
            ///
            /// This method returns no value, but logs the memory usage of the
            /// inverted index to the console, including internal structures.
            ///
            /// Example:
            ///     >>> index.print_space_usage_byte()
            ///     Total space usage: 8.4 MB
            pub fn print_space_usage_byte(&self) {
                self.inverted_index.print_space_usage_byte();
            }

            /// Get the sparse vector representation of a document by its ID.
            ///
            /// This method returns the list of token IDs and their corresponding
            /// values for the specified document in the raw index.
            ///
            /// Args:
            ///     id (int): The document ID.
            ///
            /// Returns:
            ///     tuple[list[int], list[float]]: A tuple containing the list of token IDs and their values.
            ///
            /// Example:
            ///     >>> tokens, values = index.get(5)
            ///     >>> print(tokens)
            ///     [2, 7, 11]
            #[pyo3(signature = (id))]
            #[pyo3(text_signature = "(self, id)")]
            pub fn get(&self, id: usize) -> PyResult<(Vec<$Key>, Vec<f32>)> {
                let entry = self.inverted_index.dataset().get(id as u64);
                Ok((
                    entry.components().to_vec(),
                    entry.values().iter().map(|v| v.to_f32()).collect(),
                ))
            }

            /// Load a previously saved raw SeismicIndex from disk.
            ///
            /// This method reads a `.index.seismic` file and restores the corresponding
            /// `SeismicIndexRaw` instance. The file must have been created using the `save()` method.
            ///
            /// Args:
            ///     index_path (str): Path to the `.index.seismic` file.
            ///
            /// Returns:
            ///     SeismicIndexRaw: The loaded raw index.
            ///
            /// Raises:
            ///     IOError: If the file is missing or cannot be deserialized.
            ///
            /// Example:
            ///     >>> index = seismic.SeismicIndexRaw.load("my_index.index.seismic")
            #[staticmethod]
            #[pyo3(signature = (index_path))]
            #[pyo3(text_signature = "(index_path)")]
            pub fn load(index_path: &str) -> PyResult<$rust_name> {
                let inverted_index: InvertedIndexBase<IndexDataset<$Key>> =
                    InvertedIndexBase::load_index(&index_path)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load index: {:?}", e)))?;

                Ok($rust_name { inverted_index })
            }

            /// Save the raw SeismicIndex to disk in binary format.
            ///
            /// This method serializes the raw inverted index and writes it to a `.index.seismic` file
            /// using the provided path prefix.
            ///
            /// Args:
            ///     path (str): The output path (without extension). A `.index.seismic` suffix is added automatically.
            ///
            /// Raises:
            ///     IOError: If writing to disk fails.
            ///
            /// Example:
            ///     >>> index.save("my_index")
            ///     # Creates a file named 'my_index.index.seismic'
            #[pyo3(signature = (path))]
            #[pyo3(text_signature = "(self, path)")]
            pub fn save(&self, path: &str) -> PyResult<()> {
                let full_path = format!("{}.index.seismic", path);
                println!("Saving ... {}", full_path);
                self.inverted_index.save_index(path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to write index to '{}': {:?}",
                        full_path, e
                    ))
                })?;

                Ok(())
            }

            /// Build and attach a KNN (k-nearest neighbor) graph to the raw index.
            ///
            /// This method computes the `nknn` nearest neighbors for each document
            /// using the raw seismic index, and stores the result internally.
            ///
            /// Args:
            ///     nknn (int): The number of neighbors to compute per document.
            ///
            /// Example:
            ///     >>> index.build_knn(10)
            #[pyo3(signature = (nknn))]
            #[pyo3(text_signature = "(self, nknn)")]
            pub fn build_knn(&mut self, nknn: usize) {
                let knn = Knn::new(&self.inverted_index, nknn);
                self.inverted_index.add_knn(knn);
            }

            /// Save the KNN graph associated with the raw index to disk.
            ///
            /// This method writes the internal KNN data to a file for later reuse.
            /// It requires that the index already has a KNN graph computed.
            ///
            /// Args:
            ///     path (str): Output path where the KNN file will be saved.
            ///
            /// Raises:
            ///     IOError: If saving the KNN graph fails.
            ///
            /// Example:
            ///     >>> index.save_knn("my_index.knn")
            #[pyo3(signature = (path))]
            #[pyo3(text_signature = "(self, path)")]
            pub fn save_knn(&self, path: &str) -> PyResult<()> {
                self.inverted_index
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

            /// Load a precomputed KNN graph from disk and attach it to the raw index.
            ///
            /// This method reads a serialized KNN graph from the specified path and attaches
            /// it to the current index. You can optionally limit the number of neighbors to load.
            ///
            /// Args:
            ///     knn_path (str): Path to the KNN file.
            ///     nknn (int, optional): Limit on the number of neighbors to load per document.
            ///
            /// Example:
            ///     >>> index.load_knn("my_index.knn")
            ///     >>> index.load_knn("my_index.knn", nknn=5)
            #[pyo3(signature = (knn_path, nknn=None))]
            #[pyo3(text_signature = "(self, knn_path, nknn=None)")]
            pub fn load_knn(&mut self, knn_path: &str, nknn: Option<usize>) {
                let knn = Knn::new_from_serialized(knn_path, nknn);
                self.inverted_index.add_knn(knn);
            }

            /// Build a raw SeismicIndex from a dataset in the Seismic inner format.
            ///
            /// This method constructs an inverted index from a file in Seismic inner format.
            /// It supports pruning, clustering, summarization, and optional KNN graph construction.
            ///
            ///
            /// Args:
            ///     input_file (str): Path to the binary `.bin` dataset file.
            ///     n_postings (int, optional): Max number of postings per token (default: 3500).
            ///     centroid_fraction (float, optional): Fraction of document used to form centroids in each list (default: 0.1).
            ///     min_cluster_size (int, optional): Minimum documents per cluster (default: 2).
            ///     summary_energy (float, optional): Energy threshold for summarization (default: 0.4).
            ///     max_fraction (float): Maximum number of document to keep in each inverted list computed as n_postings * max_fraction.
            ///     doc_cut (int): Thereshold for document approximation at building time.
            ///     nknn (int, optional): Number of KNN neighbors to compute or load (default: 0).
            ///     knn_path (str, optional): Path to a precomputed KNN file.
            ///     batched_indexing (int, optional): Optional batch size for indexing to reduce memory usage.
            ///
            /// Returns:
            ///     SeismicIndexRaw: A newly constructed raw index.
            ///
            /// Example:
            ///     >>> index = seismic.SeismicIndexRaw.build("documents.seismic.bin")
            #[staticmethod]
            #[allow(clippy::too_many_arguments)]
            #[pyo3(signature = (
                input_file,
                n_postings=3500,
                centroid_fraction=0.1,
                min_cluster_size=2,
                summary_energy=0.4,
                max_fraction=1.5,
                doc_cut=15,
                nknn=0,
                knn_path=None,
                batched_indexing=None
            ))]
            #[pyo3(
                text_signature = "(input_file, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, max_fraction=1.5, doc_cut=15, nknn=0, knn_path=None, batched_indexing=None)"
            )]
            pub fn build(
                input_file: &str,
                n_postings: usize,
                centroid_fraction: f32,
                min_cluster_size: usize,
                summary_energy: f32,
                max_fraction: f32,
                doc_cut: usize,

                nknn: usize,
                knn_path: Option<String>,
                batched_indexing: Option<usize>,
            ) -> PyResult<$rust_name> {
                let _ = batched_indexing;
                let dataset = vectorium::read_seismic_format::<$Key, f32, DotProduct>(input_file)
                    .unwrap();
                let dataset = quantize_f32_dataset_to_f16(dataset);

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
                println!("\nBuilding the index...");
                println!("{:?}", config);

                let inverted_index = InvertedIndexBase::build(dataset, config);
                Ok($rust_name { inverted_index })
            }

            /// Perform a nearest neighbor search on a single sparse query.
            ///
            /// This method searches the raw index using the given query, returning the top-k
            /// most similar documents based on inner product.
            ///
            /// Args:
            ///     query_components (ndarray[int32]): Array of token IDs in the query.
            ///     query_values (ndarray[float32]): Array of corresponding values.
            ///     k (int): Number of results to return.
            ///     query_cut (int): Maximum number of tokens to consider from the query.
            ///     heap_factor (float): Heap factor used during search.
            ///     n_knn (int): Number of KNN neighbors to scan during refinement.
            ///     sorted (bool, optional): Whether to scan the summaries in each posting list starting from the most similar (default: True).
            ///
            /// Returns:
            ///     list[tuple[float, int]]: A list of (score, doc_id) tuples.
            ///
            /// Example:
            ///     >>> index.search(np.array([1, 5, 7]), np.array([0.5, 0.2, 0.1]), 10, 20, 0.8, 0, True)
            #[pyo3(signature = (
                query_components,
                query_values,
                k,
                query_cut,
                heap_factor,
                n_knn,
                sorted
            ))]
            #[pyo3(
                text_signature = "(self, query_components, query_values, k, query_cut, heap_factor, n_knn, sorted)"
            )]
            #[allow(clippy::too_many_arguments)]
            pub fn search<'py>(
                &self,
                query_components: PyReadonlyArrayDyn<'py, i32>,
                query_values: PyReadonlyArrayDyn<'py, f32>,
                k: usize,
                query_cut: usize,
                heap_factor: f32,
                n_knn: usize,
                sorted: bool,
            ) -> Vec<(f32, usize)> {
                let components = query_components
                    .to_vec()
                    .unwrap()
                    .iter()
                    .map(|x| *x as $Key)
                    .collect::<Vec<_>>();
                let values = query_values.to_vec().unwrap();
                let query = SparseVectorView::new(components.as_slice(), values.as_slice());
                self.inverted_index
                    .search(
                        query,
                        k,
                        query_cut,
                        heap_factor,
                        n_knn,
                        sorted, // first_sorted is set to false
                    )
                    .into_iter()
                    .map(|result| (result.distance.distance(), result.vector as usize))
                    .collect()
            }


            /// Perform batched nearest neighbor search on multiple sparse queries.
            ///
            /// This method performs parallel search over multiple queries. Queries shall be provided
            /// in the inner Seismic format.
            ///
            /// Args:
            ///     query_path (str): Path to the binary query dataset file.
            ///     k (int): Number of results to return per query.
            ///     query_cut (int): Maximum number of tokens per query.
            ///     heap_factor (float): Heap factor used during search.
            ///     n_knn (int): Number of KNN neighbors to scan for refinement.
            ///     sorted (bool, optional): Whether to scan the summaries in each posting list starting from the most similar (default: True).
            ///     num_threads (int, optional): Number of threads for parallel execution (default: 0 = use Rayon default).
            ///
            /// Returns:
            ///     list[list[tuple[float, int]]]: A list of result lists, one per query.
            ///
            /// Example:
            ///     >>> results = index.batch_search("queries.bin", k=10, query_cut=20, heap_factor=0.8)
            #[pyo3(signature = (
                query_path,
                k,
                query_cut,
                heap_factor,
                n_knn,
                sorted,
                num_threads=0
            ))]
            #[pyo3(
                text_signature = "(self, query_path, k, query_cut, heap_factor, n_knn, sorted, num_threads=0)"
            )]
            #[allow(clippy::too_many_arguments)]
            pub fn batch_search(
                &self,
                query_path: &str,
                k: usize,
                query_cut: usize,
                heap_factor: f32,
                n_knn: usize,
                sorted: bool,
                num_threads: usize,
            ) -> Vec<Vec<(f32, usize)>> {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap();

                let queries =
                    vectorium::read_seismic_format::<$Key, f32, DotProduct>(query_path).unwrap();

                queries
                    .par_iter()
                    .map(|query| {
                        self.inverted_index
                            .search(
                                query.clone(),
                                k,
                                query_cut,
                                heap_factor,
                                n_knn,
                                sorted,
                            )
                            .into_iter()
                            .map(|result| (result.distance.distance(), result.vector as usize))
                            .collect()
                    })
                    .collect::<Vec<_>>()
            }


        }
    };
}

#[macro_use]
mod dataset; // -> import macro from pylib/dataset.rs

mod dotvbyte;
pub use dotvbyte::SeismicIndexDotVByte;

impl_seismic_dataset!(SeismicDataset, "SeismicDataset", u16);
impl_seismic_dataset!(SeismicDatasetLV, "SeismicDatasetLV", u32);

impl_seismic_index!(SeismicIndex, "SeismicIndex", u16, SeismicDataset);
impl_seismic_index!(SeismicIndexLV, "SeismicIndexLV", u32, SeismicDatasetLV);

impl_seismic_index_raw!(SeismicIndexRaw, "SeismicIndexRaw", u16);
impl_seismic_index_raw!(SeismicIndexRawLV, "SeismicIndexRawLV", u32);

//impl_seismic_index!(SeismicIndexLV, "SeismicIndexLV", u32, SeismicDatasetLV);
