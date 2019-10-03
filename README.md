# Kids_Phono_Oddball

Containing analysis scripts for a Phonological Auditory Oddball data produced by Elekta MEG scanner. 

Currently a written out script, but developing into a modular analysis script for re-usability in other projects. 

*This is a work in progress! You have been warned*

Overview
-
mne_group_level.py: 
- Contains the pipeline for bulk processing, epoching, evoked, source-recon etc

red_meg_tooks.py:
- this is a module containing functions for running different group stages.
- public functions run through a list of files.
- private functions process each of these files.
- each _multiple function has an njobs argument allowing parallel processing. 

remote_test.py
- contains a sandbox example pipeline for one participant.
- using this to construct the module and bulk script above. 

Requirements: 
-
- Python3
- MNE Python newest env
- PyQt5



