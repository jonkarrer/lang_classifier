mod data_prep;
mod training;

use crate::data_prep::gather::ClassifiedDataset;
use burn::data::dataset::Dataset;

fn main() {
    let dataset = ClassifiedDataset::training_set().unwrap().get(0).unwrap();
    dbg!(dataset);
}
