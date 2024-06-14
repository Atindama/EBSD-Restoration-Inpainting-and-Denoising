This directory contains code relevant to running the machine learning/Criminisi hybrid approach to EBSD inpainting.

## Files

- `single_predict.py`
	- Contains the code needed to use the ML model to preform an inpainting operation, assuming data is input as a numpy array in the form (height, width, channel)

- `model.py`
	- Contains code defining the ML model architecture.  Needed in order to load the model in correctly.

- `criminisi_mod.py`
	- Contains a modified version of the Criminisi inpainting code written by Wineguard.  Modifications have been made to make the `inpaint` function and a few utilities more ammendable to the hybrid method.

- `MLexemplar.py`
	- Contains the actual hybrid method code.

- `multi_test.py`
	- Contains code to automatically run a large number of tests using the hybrid inpainting methods. 

