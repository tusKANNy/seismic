macro_rules! impl_seismic_dataset {
    ($rust_name:ident, $py_name:literal, $Key:ty) => {
        /// A Python wrapper around an in-memory sparse dataset.
        ///
        /// This class provides a way to build a dataset of sparse vectors
        /// (documents) in memory before indexing them with `SeismicIndex`.
        ///
        /// Each document is represented as a set of (token, value) pairs, where
        /// tokens are strings and values are float32.
        ///
        #[derive(Clone)]
        #[pyclass(name = $py_name)]
        pub struct $rust_name {
            dataset: crate::SeismicDataset<$Key>,
        }

        #[pymethods]
        impl $rust_name {
            /// Create a new, empty SeismicDataset.
            ///
            /// This method returns an empty dataset ready to be populated
            /// with documents using `add_document()` or `add_documents_by_id()`.
            ///
            /// Args:
            ///     dim (int, optional): Number of components, when known up front
            ///         (e.g. the tokenizer vocabulary size for `add_documents_by_id()`).
            ///         Otherwise the dataset grows as new components appear, rebuilding
            ///         its storage each time.
            ///
            /// Returns:
            ///     SeismicDataset: A new, empty dataset instance.
            ///
            /// Example:
            ///     >>> dataset = seismic.SeismicDataset()
            ///     >>> dataset = seismic.SeismicDataset(dim=30522)
            #[new]
            #[pyo3(signature = (dim=0))]
            fn new(dim: usize) -> Self {
                $rust_name {
                    dataset: crate::SeismicDataset::with_dim(dim),
                }
            }

            /// Append a batch of documents given by integer component ids (CSR layout).
            ///
            /// Row `i` is `components[offsets[i]:offsets[i + 1]]` with the matching
            /// `values`; components must be strictly increasing within a row. Ids are
            /// used as they are, with no token strings: queries address component 42 as
            /// `"42"`. Rows with no components are skipped together with their `doc_id`.
            /// Much faster than one `add_document()` per row, and never rebuilds the
            /// storage when the dataset was created with `dim`. Do not mix with
            /// `add_document()`, which numbers tokens by insertion order.
            ///
            /// Args:
            ///     doc_ids (list[str]): One identifier per row.
            ///     components (ndarray[int64]): `[nnz]` component ids.
            ///     values (ndarray[float32]): `[nnz]` values.
            ///     offsets (ndarray[int64]): `[len(doc_ids) + 1]` row boundaries, starting at 0.
            ///
            /// Returns:
            ///     int: Number of documents appended (rows with no components are
            ///     not counted). On `ValueError` the dataset is left unchanged.
            ///
            /// Example:
            ///     >>> dataset.add_documents_by_id(
            ///     ...     ["d0", "d1"],
            ///     ...     np.array([3, 7, 7], dtype=np.int64),
            ///     ...     np.array([0.5, 1.0, 2.0], dtype=np.float32),
            ///     ...     np.array([0, 2, 3], dtype=np.int64),
            ///     ... )
            #[pyo3(signature = (doc_ids, components, values, offsets))]
            #[pyo3(text_signature = "(self, doc_ids, components, values, offsets)")]
            pub fn add_documents_by_id(
                &mut self,
                doc_ids: Vec<String>,
                components: PyReadonlyArray1<'_, i64>,
                values: PyReadonlyArray1<'_, f32>,
                offsets: PyReadonlyArray1<'_, i64>,
            ) -> PyResult<usize> {
                fn to_usize(name: &str, array: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
                    array
                        .as_slice()?
                        .iter()
                        .map(|&x| {
                            usize::try_from(x).map_err(|_| {
                                PyValueError::new_err(format!("{} contains a negative entry", name))
                            })
                        })
                        .collect()
                }
                let components = to_usize("components", components)?;
                let offsets = to_usize("offsets", offsets)?;
                self.dataset
                    .add_documents_by_id(doc_ids, &components, values.as_slice()?, &offsets)
                    .map_err(PyValueError::new_err)
            }

            /// Get the number of documents currently stored in the dataset.
            ///
            /// This method returns the total number of documents (sparse vectors)
            /// that have been added so far using `add_document()`.
            ///
            /// Returns:
            ///     int: The number of documents in the dataset.
            ///
            /// Example:
            ///     >>> dataset.len
            ///     3
            #[getter]
            pub fn get_len(&self) -> PyResult<usize> {
                Ok(self.dataset.len())
            }

            /// Add a sparse document to the dataset.
            ///
            /// This method adds a document identified by `doc_id`, using a list of
            /// tokens and their corresponding values. Each document is stored as a
            /// sparse vector of (token, value) pairs.
            ///
            /// Args:
            ///     doc_id (str): A unique identifier for the document.
            ///     tokens (ndarray[str]): Array of tokens (as strings).
            ///     values (ndarray[float32]): Array of corresponding values (same length as tokens).
            ///
            /// Example:
            ///     >>> string_type = seismic.get_seismic_string()
            ///     >>> dataset.add_document("doc42", np.array(["a", "b"], dtype=string_type), np.array([0.5, 1.0]))
            #[pyo3(signature = (doc_id, tokens, values))]
            #[pyo3(text_signature = "(self, doc_id, tokens, values)")]
            pub fn add_document(
                &mut self,
                doc_id: &str,
                tokens: PyReadonlyArrayDyn<'_, PyFixedUnicode<MAX_TOKEN_LEN>>,
                values: PyReadonlyArrayDyn<'_, f32>,
            ) {
                self.dataset.add_document(
                    doc_id.to_string(),
                    &tokens
                        .to_vec()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<String>>(),
                    &values.to_vec().unwrap(),
                    None,
                );
            }

            /// Perform a single sparse query search on the dataset.
            ///
            /// This method performs brute-force search over the in-memory dataset
            /// (without using an index) to find the top-k most similar documents.
            ///
            /// Args:
            ///     query_id (str): Identifier for the query (used for result annotation).
            ///     query_components (ndarray[str]): Array of tokens (components) in the query.
            ///     query_values (ndarray[float32]): Array of corresponding values for each token.
            ///     k (int): Number of results to return.
            ///
            /// Returns:
            ///     list[tuple[str, float, str]]: A list of (query_id, distance, document_id) tuples.
            ///
            /// Example:
            ///     >>> string_type = seismic.get_seismic_string()
            ///     >>> results = dataset.search("q1", np.array(["token1", "token2"], dtype=string_type), np.array([0.5, 0.3], dtype=np.float32), k=5)
            #[pyo3(signature = (query_id, query_components, query_values, k))]
            #[pyo3(text_signature = "(self, query_id, query_components, query_values, k)")]
            pub fn search<'py>(
                &self,
                query_id: String,
                query_components: PyReadonlyArrayDyn<'py, PyFixedUnicode<MAX_TOKEN_LEN>>,
                query_values: PyReadonlyArrayDyn<'py, f32>,
                k: usize,
            ) -> Vec<(String, f32, String)> {
                self.dataset.search(
                    &query_id,
                    &query_components
                        .to_vec()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>(),
                    &query_values.to_vec().unwrap(),
                    k,
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
            ///     str | None: The document text if content was stored, otherwise None.
            ///
            /// Example:
            ///     >>> text = dataset.get_doc_text("doc42")
            #[pyo3(signature = (doc_id))]
            #[pyo3(text_signature = "(self, doc_id)")]
            pub fn get_doc_text(&self, doc_id: &str) -> Option<String> {
                self.dataset.get_doc_text(doc_id)
            }


            /// Perform batched search over multiple sparse queries on the dataset.
            ///
            /// This method performs parallel search over multiple queries using brute-force
            /// computation on the in-memory dataset (without using an index).
            ///
            /// Args:
            ///     queries_ids (ndarray[str]): Array of query IDs, one per query.
            ///     query_components (list[ndarray[str]]): List of arrays, each containing the tokens of a query.
            ///     query_values (list[ndarray[float32]]): List of arrays, each containing values corresponding to tokens.
            ///     k (int): Number of results to return per query.
            ///     num_threads (int, optional): Number of threads to use for parallel execution (default: 0 = Rayon default).
            ///
            /// Returns:
            ///     list[list[tuple[str, float, str]]]: A list of result lists, one per query. Each result is a (query_id, distance, document_id) tuple.
            ///
            /// Example:
            ///     >>> results = dataset.batch_search(query_ids, query_components, query_values, k=10)
            #[pyo3(signature = (
                queries_ids,
                query_components,
                query_values,
                k,
                num_threads=0
            ))]
            #[pyo3(
                text_signature = "(self, queries_ids, query_components, query_values, k, num_threads=0)"
            )]
            pub fn batch_search<'py>(
                &self,
                queries_ids: PyReadonlyArrayDyn<'py, PyFixedUnicode<MAX_TOKEN_LEN>>,
                query_components: Bound<'py, PyList>,
                query_values: Bound<'_, PyList>,
                k: usize,
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
                        self.dataset.search(
                            &query_id.to_string(),
                            components,
                            values,
                            k,
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
