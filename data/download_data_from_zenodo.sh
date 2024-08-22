#!/bin/bash

# Download and unzip
curl https://zenodo.org/records/13362692/files/Zenodo.zip --output "Zenodo.zip"
unzip Zenodo.zip

# Move input data to ../input/
mv Zenodo/BBH_semianalytic.npy ../input/
mv Zenodo/BNS_semianalytic.npy ../input/
mv Zenodo/NSBH_semianalytic.npy ../input/
mv Zenodo/endo3_bbhpop-LIGO-T2100113-v12.hdf5 ../input/
mv Zenodo/endo3_bnspop-LIGO-T2100113-v12.hdf5 ../input/
mv Zenodo/endo3_nsbhpop-LIGO-T2100113-v12.hdf5 ../input/

# Move popsummary results to top level data/ folder
mv Zenodo/popsummary_standardInjections.h5 .
mv Zenodo/popsummary_dynamicInjections.h5 .
mv Zenodo/trained_networks/ .

# Remove original zip files and annoying Mac OSX files
rm Zenodo/.DS_Store
rm __MACOSX/Zenodo/._.DS_Store
rm __MACOSX/Zenodo/._endo3_bbhpop-LIGO-T2100113-v12.hdf5
rm __MACOSX/Zenodo/._endo3_bnspop-LIGO-T2100113-v12.hdf5
rm __MACOSX/Zenodo/._endo3_nsbhpop-LIGO-T2100113-v12.hdf5
rmdir __MACOSX/Zenodo
rmdir __MACOSX
rmdir Zenodo
rm Zenodo.zip
