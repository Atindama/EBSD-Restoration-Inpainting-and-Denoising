# Hybrid_EBSD_Restoration
This is made up of an inpainting algorithm and a denoising algorithm for EBSD data

Install the environment package in environment.yml. The environment is called hybrid-inpainting

The file create_examples_paper.py can be used to run a simple example using;

python create_examples_paper.py @ ../synthetic_EBSD_data/val im_oneblock 20 ./epoch_may3.pt cuda:0 --n-examples 1 --patch-size 3 --missing-region-location 'center'
to run the file and give the input variables. You will also have to update the source file in the create_examples_paper.py file

# sbatch create_examples.sh ../synthetic_EBSD_data/val im_oneblock 20 ./epoch_may3.pt cuda:0 --n-examples 1 --patch-size 3 --missing-region-location 'center'

# sbatch multi_test.sh ./epoch_996.pt 3 cuda:0 ./errors_CriminisiML/

Download the pretrained models here https://drive.google.com/drive/folders/1alwtFbyjpd8Y5dfscb2mmUX4Hb38WYux?usp=sharing
