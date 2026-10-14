use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader},
    time::Instant,
};

use crate::json_utils::{JsonSparseVector, extract_jsonl};
use half::f16;
use num_traits::{FromPrimitive, ToPrimitive};
use vectorium::ComponentType;
use vectorium::IndexSerializer;
use vectorium::dataset::ConvertFrom;
use vectorium::dataset::ScoredVector;
use vectorium::vector_encoder::{SparseDataEncoder, SparseVectorEncoder};
use vectorium::{
    Dataset, DatasetGrowable, Distance, DotProduct, FlatIndex, Index, QueryEvaluator,
    ScalarSparseQuantizer, SpaceUsage, SparseData, SparseDataset, SparseDatasetGrowable,
    SparseVectorView, VectorEncoder,
};

use indicatif::ProgressIterator;
use itertools::Itertools;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use serde_json::Deserializer;

use flate2::read::GzDecoder;
use tar::Archive;

use crate::{
    InvertedIndexBase,
    configurations::Configuration,
    index_traits::{ComponentFor, EncoderFor, SeismicBuildDataset, SeismicSearchDataset, ValueFor},
    inverted_index::Knn,
};

type ScoredVectorDotProduct = ScoredVector<DotProduct>;

/// Represents a single search result with query metadata, score, and document ID.
/// To retrieve document text, use `get_doc_text` on the index or dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub query_id: String,
    pub score: f32,
    pub doc_id: String,
}

impl SearchResult {
    pub fn to_tuple(&self) -> (String, f32, String) {
        (self.query_id.clone(), self.score, self.doc_id.clone())
    }
}

/// Remap raw search results (score, internal_doc_id) into `SearchResult` values
/// using the document ID mapping.
fn remap_results(
    results: impl IntoIterator<Item = (f32, usize)>,
    query_id: &str,
    doc_mapping: &[String],
) -> Vec<SearchResult> {
    results
        .into_iter()
        .map(|(score, internal_id)| SearchResult {
            query_id: query_id.to_owned(),
            score,
            doc_id: doc_mapping[internal_id].clone(),
        })
        .collect()
}

/// Resolve query token strings to internal component IDs using the token map.
/// Unknown tokens are silently filtered out. Results are sorted by component ID.
pub(crate) fn resolve_query_tokens<C: ComponentType + FromPrimitive>(
    tokens: &[String],
    values: &[f32],
    token_map: &HashMap<String, usize>,
) -> (Vec<C>, Vec<f32>) {
    tokens
        .iter()
        .zip(values)
        .filter_map(|(token, &value)| {
            token_map
                .get(token)
                .and_then(|id| C::from_usize(*id))
                .map(|component| (component, value))
        })
        .sorted_by_key(|(component, _)| *component)
        .unzip()
}

#[derive(Default, PartialEq, Clone, Serialize, Deserialize)]
pub struct SeismicIndex<S>
where
    S: SparseData,
    EncoderFor<S>: SparseDataEncoder,
{
    #[serde(bound(
        serialize = "S: Serialize, ComponentFor<S>: Serialize",
        deserialize = "S: DeserializeOwned, ComponentFor<S>: DeserializeOwned"
    ))]
    inverted_index: InvertedIndexBase<S>,
    document_mapping: Option<Box<[String]>>,
    document_content: Option<Box<[Option<String>]>>,
    token_to_id_map: HashMap<String, usize>,
}

impl<S> IndexSerializer for SeismicIndex<S>
where
    S: SparseData,
    EncoderFor<S>: SparseDataEncoder,
{
}

impl<S> SpaceUsage for SeismicIndex<S>
where
    S: Dataset + SparseData + SpaceUsage,
    EncoderFor<S>: SparseDataEncoder,
    ComponentFor<S>: SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        let mut total = self.inverted_index.space_usage_bytes();

        // document_mapping: Box<[String]> overhead + string contents
        if let Some(ref mapping) = self.document_mapping {
            total += std::mem::size_of::<String>() * mapping.len();
            total += mapping.iter().map(|s| s.capacity()).sum::<usize>();
        }

        // document_content: Box<[Option<String>]> overhead + string contents
        if let Some(ref content) = self.document_content {
            total += std::mem::size_of::<Option<String>>() * content.len();
            total += content
                .iter()
                .filter_map(|s| s.as_ref())
                .map(|s| s.capacity())
                .sum::<usize>();
        }

        // token_to_id_map: HashMap overhead + key string contents
        // Approximate HashMap overhead: each entry has a key (String) + value (usize) + hash metadata
        total += self.token_to_id_map.capacity()
            * (std::mem::size_of::<String>() + std::mem::size_of::<usize>() + 8);
        total += self
            .token_to_id_map
            .keys()
            .map(|k| k.capacity())
            .sum::<usize>();

        total
    }
}

// Build-only methods: require SeismicBuildDataset + SeismicSearchDataset
impl<S> SeismicIndex<S>
where
    S: SeismicBuildDataset + SeismicSearchDataset,
    EncoderFor<S>: SparseVectorEncoder<InputValueType = f32>,
    S: From<SparseDataset<EncoderFor<S>>>,
{
    pub fn new(
        dataset: S,
        config: Configuration,
        document_mapping: Option<impl Into<Box<[String]>>>,
        document_content: Option<impl Into<Box<[Option<String>]>>>,
        token_to_id_map: HashMap<String, usize>,
    ) -> Self
    where
        ValueFor<S>: vectorium::FromF32,
        for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
            From<SparseVectorView<'a, ComponentFor<S>, f32>>,
        for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>: QueryEvaluator<
                <EncoderFor<S> as VectorEncoder>::EncodedVector<'a>,
                Distance = DotProduct,
            >,
    {
        let inverted_index = InvertedIndexBase::build(dataset, config);

        Self {
            inverted_index,
            document_mapping: document_mapping.map(|d| d.into()),
            document_content: document_content.map(|c| c.into()),
            token_to_id_map,
        }
    }
}

// Search and access methods: only require SeismicSearchDataset
impl<S> SeismicIndex<S>
where
    S: SeismicSearchDataset,
{
    pub fn remap_doc_ids(
        &self,
        plain_results: impl IntoIterator<Item = ScoredVectorDotProduct>,
        query_id: &str,
    ) -> Vec<SearchResult> {
        let score_id_pairs: Vec<(f32, usize)> = plain_results
            .into_iter()
            .map(|result| (result.distance.distance(), result.vector as usize))
            .collect();

        match &self.document_mapping {
            Some(mapping) => remap_results(score_id_pairs, query_id, mapping),
            None => score_id_pairs
                .into_iter()
                .map(|(score, internal_id)| SearchResult {
                    query_id: query_id.to_owned(),
                    score,
                    doc_id: internal_id.to_string(),
                })
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_raw(
        &self,
        query_components_original: &[String],
        query_values: &[f32],
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        first_sorted: bool,
    ) -> Vec<ScoredVectorDotProduct>
    where
        ComponentFor<S>: FromPrimitive,
        for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
            From<SparseVectorView<'a, ComponentFor<S>, f32>>,
        for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>: QueryEvaluator<
                <EncoderFor<S> as VectorEncoder>::EncodedVector<'a>,
                Distance = DotProduct,
            >,
    {
        let (filtered_query_components, filtered_query_values) =
            resolve_query_tokens::<ComponentFor<S>>(
                query_components_original,
                query_values,
                &self.token_to_id_map,
            );

        let query = SparseVectorView::new(
            filtered_query_components.as_slice(),
            filtered_query_values.as_slice(),
        );
        self.inverted_index
            .search(query, k, query_cut, heap_factor, n_knn, first_sorted)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        query_id: &str,
        query_components_original: &[String],
        query_values: &[f32],
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        first_sorted: bool,
    ) -> Vec<SearchResult>
    where
        ComponentFor<S>: FromPrimitive,
        for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
            From<SparseVectorView<'a, ComponentFor<S>, f32>>,
        for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>: QueryEvaluator<
                <EncoderFor<S> as VectorEncoder>::EncodedVector<'a>,
                Distance = DotProduct,
            >,
    {
        let results = self.search_raw(
            query_components_original,
            query_values,
            k,
            query_cut,
            heap_factor,
            n_knn,
            first_sorted,
        );

        self.remap_doc_ids(results, query_id)
    }

    /// Look up the stored text content for a document by its string ID.
    /// Returns `None` if content was not loaded or the doc ID is not found.
    pub fn get_doc_text(&self, doc_id: &str) -> Option<String> {
        let mapping = self.document_mapping.as_deref()?;
        let content = self.document_content.as_deref()?;
        let idx = mapping.iter().position(|id| id == doc_id)?;
        content[idx].clone()
    }
}

// Accessor methods with minimal trait bounds
impl<S> SeismicIndex<S>
where
    S: SparseData,
    EncoderFor<S>: SparseDataEncoder,
{
    pub fn get_doc_ids_in_postings(&self, list_id: usize) -> Vec<usize> {
        self.inverted_index.get_doc_ids_in_postings(list_id)
    }

    pub fn print_space_usage_byte(&self)
    where
        S: SpaceUsage,
        ComponentFor<S>: SpaceUsage,
    {
        self.inverted_index.print_space_usage_byte();
    }

    pub fn dim(&self) -> usize {
        self.inverted_index.dim()
    }

    pub fn nnz(&self) -> usize {
        self.inverted_index.nnz()
    }

    pub fn len(&self) -> usize {
        self.inverted_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inverted_index.is_empty()
    }

    pub fn inverted_index(&self) -> &InvertedIndexBase<S> {
        &self.inverted_index
    }

    pub fn dataset(&self) -> &S {
        self.inverted_index.dataset()
    }

    pub fn convert_dataset_into<T>(self) -> SeismicIndex<T>
    where
        T: Dataset + SparseData + ConvertFrom<S> + SpaceUsage,
        EncoderFor<T>: SparseDataEncoder<OutputComponentType = ComponentFor<S>>,
        ComponentFor<T>: SpaceUsage,
    {
        SeismicIndex {
            inverted_index: self.inverted_index.convert_dataset_into(),
            document_mapping: self.document_mapping,
            document_content: self.document_content,
            token_to_id_map: self.token_to_id_map,
        }
    }

    pub fn add_knn(&mut self, knn: Knn) {
        self.inverted_index.add_knn(knn);
    }

    pub fn knn_len(&self) -> usize {
        self.inverted_index.knn_len()
    }
}

type ScalarSparseDataset<C, V> = SparseDataset<ScalarSparseQuantizer<C, f32, V, DotProduct>>;

impl<C, V> SeismicIndex<ScalarSparseDataset<C, V>>
where
    C: ComponentType + FromPrimitive + std::hash::Hash,
    V: vectorium::ValueType + vectorium::Float + vectorium::FromF32,
{
    pub fn from_dataset(dataset: SeismicDataset<C>, config: Configuration) -> Self
    where
        for<'a> <ScalarSparseQuantizer<C, f32, V, DotProduct> as VectorEncoder>::InputVector<'a>:
            From<SparseVectorView<'a, C, f32>>,
    {
        let dim = dataset.sparse_dataset.input_dim();
        let encoder = ScalarSparseQuantizer::<C, f32, V, DotProduct>::new(dim, dim);
        let mut converted =
            SparseDatasetGrowable::<ScalarSparseQuantizer<C, f32, V, DotProduct>>::new(encoder);

        for vec in dataset.sparse_dataset.iter() {
            let components = vec.components().to_vec();
            let values_f32: Vec<f32> = vec.values().iter().map(|v| v.to_f32().unwrap()).collect();
            let input_vec: <ScalarSparseQuantizer<C, f32, V, DotProduct> as VectorEncoder>::InputVector<'_> =
                    SparseVectorView::new(components.as_slice(), values_f32.as_slice());
            converted.push(input_vec);
        }

        let frozen: ScalarSparseDataset<C, V> = converted.into();
        Self::new(
            frozen,
            config,
            Some(dataset.document_mapping),
            Some(dataset.document_content),
            dataset.token_to_id_map,
        )
    }

    /// Build the token-to-ID mapping by scanning the stream once.
    /// Also counts the number of rows, eliminating the need for a separate line-counting pass.
    fn build_token_map(reader: BufReader<impl std::io::Read>) -> (HashMap<String, usize>, usize) {
        let mut token_to_id_mapping = HashMap::<String, usize>::new();
        let mut row_count = 0usize;
        let stream: serde_json::StreamDeserializer<_, JsonSparseVector> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream {
            row_count += 1;
            let (_doc_id, tokens, _values, _content) = extract_jsonl::<f32>(x.unwrap());
            for token in tokens {
                if !token_to_id_mapping.contains_key(&token) {
                    token_to_id_mapping.insert(token, token_to_id_mapping.len());
                }
            }
        }

        let n_bits = size_of::<C>() as u32 * 8;
        assert!(
            token_to_id_mapping.len() < 2_usize.pow(n_bits),
            "The number of different tokens exceeds 2^{}.",
            n_bits
        );

        (token_to_id_mapping, row_count)
    }

    #[allow(clippy::type_complexity)]
    fn process_data(
        reader: BufReader<impl std::io::Read>,
        row_count: usize,
        token_to_id_mapping: HashMap<String, usize>,
        load_content: bool,
    ) -> (
        ScalarSparseDataset<C, V>,
        Vec<String>,
        Option<Vec<Option<String>>>,
        HashMap<String, usize>,
    ) {
        let mut doc_id_mapping = Vec::with_capacity(row_count);
        let mut doc_content_mapping = if load_content {
            Some(Vec::with_capacity(row_count))
        } else {
            None
        };
        let dim = token_to_id_mapping.len();
        let encoder = ScalarSparseQuantizer::<C, f32, V, DotProduct>::new(dim, dim);
        let mut converted_data =
            SparseDatasetGrowable::<ScalarSparseQuantizer<C, f32, V, DotProduct>>::new(encoder);

        let stream: serde_json::StreamDeserializer<_, JsonSparseVector> =
            Deserializer::from_reader(reader).into_iter();

        for x in stream.into_iter().progress_count(row_count as u64) {
            let (doc_id, tokens, values, content) = extract_jsonl::<f32>(x.unwrap());
            doc_id_mapping.push(doc_id);
            if let Some(ref mut mapping) = doc_content_mapping {
                mapping.push(content);
            }

            let ids: Vec<_> = tokens
                .into_iter()
                .map(|t| {
                    C::from_usize(token_to_id_mapping[&t])
                        .expect("Failed to convert token id to component type")
                })
                .collect();

            let converted_iterator = ids.into_iter().zip(values).sorted_by_key(|&(c, _)| c);
            let (components, values): (Vec<_>, Vec<_>) = converted_iterator.unzip();
            converted_data.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }

        let frozen: ScalarSparseDataset<C, V> = converted_data.into();
        (
            frozen,
            doc_id_mapping,
            doc_content_mapping,
            token_to_id_mapping,
        )
    }

    /// Core loading logic shared by `from_json` and `from_tar`.
    /// Takes a closure that creates a fresh `BufReader` each time it is called
    /// (needed for the 2-pass approach: first pass builds token map + counts rows,
    /// second pass processes data).
    fn from_reader_factory(
        make_reader: impl Fn() -> BufReader<Box<dyn io::Read>>,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
        load_content: bool,
    ) -> Self {
        println!("Reading the collection..");
        let start = Instant::now();

        let (token_to_id_mapping, row_count) = match input_token_to_id_map {
            Some(mapping) => {
                // Still need row count: do a single pass just counting
                let reader = make_reader();
                let stream: serde_json::StreamDeserializer<_, JsonSparseVector> =
                    Deserializer::from_reader(reader).into_iter();
                let count = stream.count();
                (mapping, count)
            }
            None => Self::build_token_map(make_reader()),
        };
        println!("Number of rows: {}", row_count);

        let reader = make_reader();
        let (final_data, doc_id_mapping, doc_content_mapping, token_to_id_mapping) =
            Self::process_data(reader, row_count, token_to_id_mapping, load_content);

        println!(
            "Elapsed time to read the collection: {:} secs",
            start.elapsed().as_secs()
        );

        Self::new(
            final_data,
            config,
            Some(doc_id_mapping),
            doc_content_mapping,
            token_to_id_mapping,
        )
    }

    pub fn from_file(
        file_path: &str,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
        load_content: bool,
    ) -> Result<Self, io::Error> {
        if file_path.ends_with(".jsonl") {
            Ok(Self::from_json(
                file_path,
                config,
                input_token_to_id_map,
                load_content,
            ))
        } else if file_path.ends_with(".tar.gz") {
            Ok(Self::from_tar(
                file_path,
                config,
                input_token_to_id_map,
                load_content,
            ))
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Unsupported file type. Supported files: .jsonl, .tar.gz",
            ))
        }
    }

    pub fn from_json(
        json_path: &str,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
        load_content: bool,
    ) -> Self {
        let json_path = json_path.to_owned();
        Self::from_reader_factory(
            move || {
                let f = File::open(&json_path)
                    .unwrap_or_else(|_| panic!("Unable to open {}", json_path));
                BufReader::new(Box::new(f) as Box<dyn io::Read>)
            },
            config,
            input_token_to_id_map,
            load_content,
        )
    }

    pub fn from_tar(
        tar_path: &str,
        config: Configuration,
        input_token_to_id_map: Option<HashMap<String, usize>>,
        load_content: bool,
    ) -> Self {
        let tar_path = tar_path.to_owned();
        Self::from_reader_factory(
            move || {
                let tar_gz_file =
                    File::open(&tar_path).unwrap_or_else(|_| panic!("Unable to open {}", tar_path));
                let gz_decoder = GzDecoder::new(tar_gz_file);
                let mut archive = Archive::new(gz_decoder);
                let json_entry = archive.entries().unwrap().next().unwrap().unwrap();
                // Read the entire entry into memory so we don't borrow from the archive
                let mut buf = Vec::new();
                io::Read::read_to_end(&mut BufReader::new(json_entry), &mut buf).unwrap();
                BufReader::new(Box::new(io::Cursor::new(buf)) as Box<dyn io::Read>)
            },
            config,
            input_token_to_id_map,
            load_content,
        )
    }
}

#[derive(Debug, Clone)]
pub struct SeismicDataset<C>
where
    C: ComponentType,
{
    sparse_dataset: vectorium::PlainSparseDatasetGrowable<C, f16, vectorium::DotProduct>,
    document_mapping: Vec<String>,
    document_content: Vec<Option<String>>,
    token_to_id_map: HashMap<String, usize>,
}

impl<C> Default for SeismicDataset<C>
where
    C: ComponentType + std::convert::TryFrom<usize> + FromPrimitive,
    <C as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C> SeismicDataset<C>
where
    C: ComponentType + std::convert::TryFrom<usize> + FromPrimitive,
    <C as std::convert::TryFrom<usize>>::Error: std::fmt::Debug,
{
    pub fn new() -> Self {
        Self::with_dim(0)
    }

    /// Empty dataset over `dim` components. Knowing the dimension up front
    /// (e.g. the tokenizer vocabulary size) avoids the storage rebuild that
    /// `ensure_dim` does every time a larger component shows up.
    pub fn with_dim(dim: usize) -> Self {
        let sparse_dataset = {
            let quantizer =
                vectorium::PlainSparseQuantizer::<C, f16, vectorium::DotProduct>::new(dim, dim);
            vectorium::PlainSparseDatasetGrowable::<C, f16, vectorium::DotProduct>::new(quantizer)
        };
        Self {
            sparse_dataset,
            document_mapping: Vec::new(),
            document_content: Vec::new(),
            token_to_id_map: HashMap::new(),
        }
    }

    pub fn sparse_dataset(
        self,
    ) -> vectorium::PlainSparseDatasetGrowable<C, f16, vectorium::DotProduct> {
        self.sparse_dataset
    }

    pub fn token_to_id_map(&self) -> &HashMap<String, usize> {
        &self.token_to_id_map
    }

    pub fn len(&self) -> usize {
        self.sparse_dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sparse_dataset.is_empty()
    }

    fn ensure_dim(&mut self, dim: usize) {
        if dim <= self.sparse_dataset.input_dim() {
            return;
        }

        let mut rebuilt = {
            let quantizer =
                vectorium::PlainSparseQuantizer::<C, f16, vectorium::DotProduct>::new(dim, dim);
            vectorium::PlainSparseDatasetGrowable::<C, f16, vectorium::DotProduct>::new(quantizer)
        };

        for vec in self.sparse_dataset.iter() {
            let components = vec.components().to_vec();
            let values = vec.values().to_vec();
            rebuilt.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }

        self.sparse_dataset = rebuilt;
    }

    pub fn add_document(
        &mut self,
        id: String,
        tokens: &[String],
        values: &[f32],
        content: Option<String>,
    ) {
        self.document_mapping.push(id);
        self.document_content.push(content);

        let mut components: Vec<C> = Vec::with_capacity(tokens.len());
        for c in tokens.iter() {
            let next_token_id = self.token_to_id_map.len();
            let token_id = *self
                .token_to_id_map
                .entry(c.to_string())
                .or_insert_with(|| next_token_id);
            components.push(C::try_from(token_id).unwrap());
            let n_bits = size_of::<C>() as u32 * 8;
            assert!(
                self.token_to_id_map.len() < 2_usize.pow(n_bits),
                "The number of different tokens exceeds 2^{}.",
                n_bits
            );
        }

        self.ensure_dim(self.token_to_id_map.len());

        let values: Vec<_> = values.iter().map(|v| f16::from_f32(*v)).collect();
        let sorted_components_values = components
            .into_iter()
            .zip(values)
            .sorted_by_key(|(c, _)| *c);
        let (components, values): (Vec<_>, Vec<_>) = sorted_components_values.unzip();

        self.sparse_dataset.push(SparseVectorView::new(
            components.as_slice(),
            values.as_slice(),
        ));
    }

    /// Append a batch of documents given directly by component id, in CSR
    /// layout: row `i` is `components[offsets[i]..offsets[i + 1]]` and the
    /// matching `values`, with components strictly increasing. Ids are stored
    /// as they are; the token map gets the identity entry `"42" -> 42` for each
    /// new id, so string queries keep resolving. Rows without components are
    /// skipped together with their id (they can never score); the returned
    /// count is the number of rows actually appended. The whole batch is
    /// validated before anything is pushed, so on `Err` the dataset is
    /// unchanged. Not meant to be mixed with `add_document`, which assigns
    /// ids by insertion order.
    pub fn add_documents_by_id(
        &mut self,
        ids: Vec<String>,
        components: &[usize],
        values: &[f32],
        offsets: &[usize],
    ) -> Result<usize, String> {
        if offsets.len() != ids.len() + 1 || offsets.first() != Some(&0) {
            return Err(format!(
                "expected {} offsets starting at 0, got {}",
                ids.len() + 1,
                offsets.len()
            ));
        }
        if components.len() != values.len() || offsets.last() != Some(&components.len()) {
            return Err(format!(
                "{} components, {} values, offsets end at {:?}",
                components.len(),
                values.len(),
                offsets.last()
            ));
        }
        let dim = components.iter().max().map_or(0, |c| c + 1);
        if dim > 0 && C::try_from(dim - 1).is_err() {
            return Err(format!("component {} exceeds the component type", dim - 1));
        }
        // Validate every row before touching the dataset.
        for (row, id) in offsets.windows(2).zip(ids.iter()) {
            let (start, end) = (row[0], row[1]);
            if start > end || end > components.len() {
                return Err(format!("offsets are not non-decreasing at {}", start));
            }
            if !components[start..end].windows(2).all(|w| w[0] < w[1]) {
                return Err(format!(
                    "components of document {} are not strictly increasing",
                    id
                ));
            }
        }

        self.ensure_dim(dim);

        let mut new_ids: Vec<usize> = components.to_vec();
        new_ids.sort_unstable();
        new_ids.dedup();
        for c in new_ids {
            self.token_to_id_map.entry(c.to_string()).or_insert(c);
        }

        let mut inserted = 0;
        for (row, id) in offsets.windows(2).zip(ids) {
            let (start, end) = (row[0], row[1]);
            if start == end {
                continue;
            }
            let row_components: Vec<C> = components[start..end]
                .iter()
                .map(|&c| C::try_from(c).unwrap())
                .collect();
            let row_values: Vec<f16> = values[start..end]
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect();
            self.sparse_dataset.push(SparseVectorView::new(
                row_components.as_slice(),
                row_values.as_slice(),
            ));
            self.document_mapping.push(id);
            self.document_content.push(None);
            inserted += 1;
        }
        Ok(inserted)
    }

    pub fn search(
        &self,
        query_id: &str,
        query_components: &[String],
        query_values: &[f32],
        k: usize,
    ) -> Vec<SearchResult> {
        let (components, values) =
            resolve_query_tokens::<C>(query_components, query_values, &self.token_to_id_map);
        let query = SparseVectorView::new(components.as_slice(), values.as_slice());
        // vectorium 0.4 removed `Dataset::search`; brute-force search now goes
        // through `FlatIndex`. `&Dataset` implements `Dataset`, so we wrap a
        // borrow of the dataset to avoid cloning it on every query.
        let flat_index = FlatIndex::from(&self.sparse_dataset);
        let plain_results: Vec<(f32, usize)> = flat_index
            .search(query, k, &())
            .into_iter()
            .map(|result| (result.distance.distance(), result.vector as usize))
            .collect();

        self.remap_doc_ids(plain_results, query_id)
    }

    pub fn remap_doc_ids(
        &self,
        plain_results: Vec<(f32, usize)>,
        query_id: &str,
    ) -> Vec<SearchResult> {
        remap_results(plain_results, query_id, &self.document_mapping)
    }

    /// Look up the stored text content for a document by its string ID.
    /// Returns `None` if content was not stored for this document or the doc ID is not found.
    pub fn get_doc_text(&self, doc_id: &str) -> Option<String> {
        let idx = self.document_mapping.iter().position(|id| id == doc_id)?;
        self.document_content[idx].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(results: &[SearchResult]) -> Vec<(String, f32)> {
        results
            .iter()
            .map(|r| (r.doc_id.clone(), r.score))
            .collect()
    }

    #[test]
    fn add_documents_by_id_matches_add_document() {
        let mut by_token = SeismicDataset::<u16>::new();
        by_token.add_document("d0".into(), &["1".into(), "7".into()], &[0.5, 1.0], None);
        by_token.add_document("d2".into(), &["3".into(), "7".into()], &[2.0, 0.25], None);

        // Same two docs in one CSR call, plus an empty row ("d1") that is skipped.
        let mut by_id = SeismicDataset::<u16>::with_dim(8);
        let inserted = by_id
            .add_documents_by_id(
                vec!["d0".into(), "d1".into(), "d2".into()],
                &[1, 7, 3, 7],
                &[0.5, 1.0, 2.0, 0.25],
                &[0, 2, 2, 4],
            )
            .unwrap();
        assert_eq!(inserted, 2);
        assert_eq!(by_id.len(), 2);

        let query: Vec<String> = vec!["7".into(), "3".into()];
        let expected = ranked(&by_token.search("q", &query, &[1.0, 1.0], 2));
        let got = ranked(&by_id.search("q", &query, &[1.0, 1.0], 2));
        assert_eq!(got, expected);
        assert_eq!(got[0], ("d2".to_string(), 2.25));
    }

    #[test]
    fn add_documents_by_id_rejects_bad_input() {
        let mut dataset = SeismicDataset::<u16>::with_dim(8);
        let id = || vec!["d0".to_string()];
        // Unsorted row, offsets past the end, component beyond u16.
        assert!(
            dataset
                .add_documents_by_id(id(), &[7, 3], &[1.0, 1.0], &[0, 2])
                .is_err()
        );
        assert!(
            dataset
                .add_documents_by_id(id(), &[3], &[1.0], &[0, 2])
                .is_err()
        );
        assert!(
            dataset
                .add_documents_by_id(id(), &[70_000], &[1.0], &[0, 1])
                .is_err()
        );
        // A bad second row must not leave the first one inserted.
        assert!(
            dataset
                .add_documents_by_id(
                    vec!["d0".into(), "d1".into()],
                    &[1, 7, 7, 3],
                    &[1.0, 1.0, 1.0, 1.0],
                    &[0, 2, 4],
                )
                .is_err()
        );
        assert_eq!(dataset.len(), 0);
        assert!(dataset.token_to_id_map().is_empty());
    }
}
