# Kids_Phono_Oddball

Containing analysis scripts for a Phonological Auditory Oddball data produced by Elekta MEG scanner. 

Currently a written out script, but developing into a modular analysis script for re-usability in other projects. 

*This is a work in progress! You have been warned*

Overview
-

RedMegTools
- This is a module containing submodules to help with bulk processing.
- RedMegTools.preprocess: allows filtering and ICA denoising
- RedMegTools.epoch: both epoching and evoked response calculation
- RedMegTools.sourcespace: will contain scripts for building source models.

mne_group_level.py: 
- Contains the pipeline for bulk processing, epoching, evoked, source-recon etc

remote_test.py
- contains a sandbox example pipeline for one participant.
- using this to construct the module and bulk script above. 

Requirements: 
-
- Python3
- MNE Python newest env
- PyQt5



