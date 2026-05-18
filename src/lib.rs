#![doc = include_str!("../README.md")]

use vectorium::{DotProduct, PlainSparseDataset};
pub use vectorium::{
    ScalarSparseDataset, ScalarSparseQuantizer, SparseDataset, SparseDatasetGrowable,
};

pub mod configurations;
pub(crate) mod posting_list;

pub mod inverted_index;
pub use inverted_index::InvertedIndexBase;

pub mod index_traits;
pub use index_traits::{SeismicBuildDataset, SeismicSearchDataset};

pub type PlainInvertedIndex<C, V> =
    inverted_index::InvertedIndexBase<PlainSparseDataset<C, V, DotProduct>>;
pub type ScalarInvertedIndex<C, W, V> =
    inverted_index::InvertedIndexBase<ScalarSparseDataset<C, W, V, DotProduct>>;

pub mod inverted_index_wrapper;
pub use inverted_index_wrapper::SearchResult;
pub use inverted_index_wrapper::SeismicDataset;
pub use inverted_index_wrapper::SeismicIndex;

pub mod quantized_summary;
pub use quantized_summary::QuantizedSummary;

pub mod json_utils;
// Vectorium now provides `SparseVectorView` / `SparseVectorOwned` with typed accessors.
// Seismic uses those directly.
pub mod utils;

/// Type aliases for quantized fixed-point types (re-exported from vectorium).
pub use vectorium::{FixedU8Q, FixedU16Q};
pub use vectorium::{BlockedSparseEncoder, BlockedSparseDataset};

#[cfg(feature = "python")]
pub mod pylib;

#[cfg(feature = "python")]
use {
    pylib::SeismicDataset as PySeismicDataset,
    pylib::SeismicDatasetLV as PySeismicDatasetLV,
    pylib::SeismicIndex as PySeismicIndex,
    pylib::SeismicIndexDotVByte,
    pylib::SeismicIndexLV as PySeismicIndexLV,
    pylib::SeismicIndexRaw,
    pylib::SeismicIndexRawLV,
    pylib::get_seismic_string,
    pyo3::prelude::PyModule,
    pyo3::prelude::*,
    pyo3::{Bound, PyResult, pymodule},
};

#[cfg(feature = "python")]
/// A Python module implemented in Rust. The name of this function must match the `lib.name`
/// setting in the `Cargo.toml`, otherwise Python will not be able to import the module.
#[pymodule]
fn seismic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_seismic_string, m)?)?;
    m.add_class::<PySeismicIndex>()?;
    m.add_class::<PySeismicIndexLV>()?;
    m.add_class::<SeismicIndexRaw>()?;
    m.add_class::<SeismicIndexRawLV>()?;
    m.add_class::<SeismicIndexDotVByte>()?;
    m.add_class::<PySeismicDataset>()?;
    m.add_class::<PySeismicDatasetLV>()?;
    Ok(())
}
