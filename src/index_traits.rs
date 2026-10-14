//! Defines marker traits for Seismic dataset operations.
//!
//! # Overview
//!
//! This module provides two key traits:
//!
//! - [`SeismicBuildDataset`] — for datasets used during index **construction**
//! - [`SeismicSearchDataset`] — for datasets used during index **search**
//!
//! # Supported Configurations
//!
//! | Category | Supported Types |
//! |----------|-----------------|
//! | Component Types | `u16`, `u32` |
//! | Value Types | `f16`, `f32` |
//! | Dataset Types | `SparseDataset`, `PlainSparseDataset`, `SparseDatasetGrowable` |
//! | Quantizers | Any `ScalarQuantizer` variant |

use std::hash::Hash;

use num_traits::ToPrimitive;
use vectorium::vector_encoder::{
    PackedSparseVectorEncoder, SparseDataEncoder, SparseVectorEncoder,
};
use vectorium::{
    ComponentType, Dataset, DotProduct, FromF32, PackedSparseDataset, PackedSparseDatasetGrowable,
    SparseData, SparseDataset, SparseDatasetGrowable, ValueType, VectorEncoder,
};

// Type Aliases (Centralized)

/// The encoder type for a dataset.
pub type EncoderFor<S> = <S as Dataset>::Encoder;

/// The component type (e.g., `u16`, `u32`) for a dataset.
pub type ComponentFor<S> = <EncoderFor<S> as SparseDataEncoder>::OutputComponentType;

/// The value type (e.g., `f16`, `f32`) for a dataset.
pub type ValueFor<S> = <EncoderFor<S> as SparseDataEncoder>::OutputValueType;

// SeismicBuildDataset Marker Trait

/// Trait for datasets usable in index **construction**.
///
/// # Bounds
///
/// | Bound | Purpose |
/// |-------|---------|
/// | `Dataset` | Access `len()`, `input_dim()`, `get()` |
/// | `SparseData` | Iterate via `iter()`, range access |
/// | `Sync` | Parallel build with Rayon |
/// | `SparseVectorEncoder` | Encode input vectors |
/// | `SparseDataEncoder` | Access component/value types |
/// | `VectorEncoder<Distance = DotProduct, QueryParams = ()>` | Distance during k-means; evaluators need no query parameters |
/// | `OutputComponentType: Hash + Ord` | HashMap keys, sorted posting lists |
/// | `OutputValueType: ToPrimitive + FromF32` | Value conversion during pruning |
pub trait SeismicBuildDataset:
    Dataset<
        Encoder: SparseVectorEncoder
                     + SparseDataEncoder<
            OutputComponentType: ComponentType + Hash + Ord,
            OutputValueType: ValueType + ToPrimitive + FromF32,
        > + VectorEncoder<Distance = DotProduct, QueryParams = ()>,
    > + SparseData
    + Sync
{
}

// Impl for SparseDataset (with any encoder)
impl<E> SeismicBuildDataset for SparseDataset<E>
where
    E: SparseVectorEncoder + SparseDataEncoder + VectorEncoder<Distance = DotProduct, QueryParams = ()>,
    <E as SparseDataEncoder>::OutputComponentType: ComponentType + Hash + Ord,
    <E as SparseDataEncoder>::OutputValueType: ValueType + ToPrimitive + FromF32,
{
}

// Impl for SparseDatasetGrowable
impl<E> SeismicBuildDataset for SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder + SparseDataEncoder + VectorEncoder<Distance = DotProduct, QueryParams = ()>,
    <E as SparseDataEncoder>::OutputComponentType: ComponentType + Hash + Ord,
    <E as SparseDataEncoder>::OutputValueType: ValueType + ToPrimitive + FromF32,
{
}

// SeismicSearchDataset Marker Trait

/// Trait for datasets usable in index **search**.
///
/// # Relationship to SeismicBuildDataset
///
/// Fewer bounds than `SeismicBuildDataset`:
/// - No `ToPrimitive`/`FromF32` (values not converted during search)
///
/// # Bounds
///
/// | Bound | Purpose |
/// |-------|---------|
/// | `Dataset` | Access vectors via `get()` |
/// | `SparseData` | Range-based access |
/// | `Sync` | Parallel search |
/// | `SparseDataEncoder` | Component type for summaries |
/// | `VectorEncoder<Distance = DotProduct, QueryParams = ()>` | Distance computation; evaluators need no query parameters |
/// | `OutputComponentType: Hash + Ord` | Summary component matching |
pub trait SeismicSearchDataset:
    Dataset<
        Encoder: SparseDataEncoder<OutputComponentType: ComponentType + Hash + Ord>
                     + VectorEncoder<Distance = DotProduct, QueryParams = ()>,
    > + SparseData
    + Sync
{
}

impl<E> SeismicSearchDataset for SparseDataset<E>
where
    E: SparseVectorEncoder + SparseDataEncoder + VectorEncoder<Distance = DotProduct, QueryParams = ()>,
    <E as SparseDataEncoder>::OutputComponentType: ComponentType + Hash + Ord,
{
}

impl<E> SeismicSearchDataset for SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder + SparseDataEncoder + VectorEncoder<Distance = DotProduct, QueryParams = ()>,
    <E as SparseDataEncoder>::OutputComponentType: ComponentType + Hash + Ord,
{
}

// Impls for PackedSparseDataset (used for packed/compressed storage)
impl<E> SeismicSearchDataset for PackedSparseDataset<E>
where
    E: PackedSparseVectorEncoder + SparseDataEncoder + VectorEncoder<Distance = DotProduct, QueryParams = ()>,
    <E as SparseDataEncoder>::OutputComponentType: ComponentType + Hash + Ord,
{
}

impl<E> SeismicSearchDataset for PackedSparseDatasetGrowable<E>
where
    E: PackedSparseVectorEncoder + SparseDataEncoder + VectorEncoder<Distance = DotProduct, QueryParams = ()>,
    <E as SparseDataEncoder>::OutputComponentType: ComponentType + Hash + Ord,
{
}
