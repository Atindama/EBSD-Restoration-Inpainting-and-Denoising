# Hybrid_EBSD_Restoration
This is made up of an inpainting algorithm and a denoising algorithm for EBSD data

Install the environment package in environment.yml. The environment is called hybrid-inpainting

The file create_examples_paper.py can be used to run a simple example using;

python create_examples_paper.py @ ../synthetic_EBSD_data/val im_oneblock 20 ./epoch_may3.pt cuda:0 --n-examples 1 --patch-size 3 --missing-region-location 'center'
to run the file and give the input variables. You will also have to update the source file in the create_examples_paper.py file

sbatch create_examples.sh ../synthetic_EBSD_data/val im_oneblock 20 ./epoch_may3.pt cuda:0 --n-examples 1 --patch-size 3 --missing-region-location 'center'

sbatch multi_test.sh ./epoch_996.pt 3 cuda:0 ./errors_CriminisiML/

Download the pretrained models here https://drive.google.com/drive/folders/1alwtFbyjpd8Y5dfscb2mmUX4Hb38WYux?usp=sharing and keep in hybrid_EBSD_restoration frolder when running.

For denoising, download the ipf_Folder, and put into the pyEBSD folder when running the demo scripts
https://drive.google.com/drive/folders/1LTa24fsjtEZR6NbysCDJbJiRkHwFIjVU?usp=sharing


![ipf_clean](https://github.com/Atindama/EBSD-Restoration-Inpainting-and-Denoising/assets/121004801/08d136b9-0c6b-423e-9dc4-2a706d9cf2b2)
![ipf_original](https://github.com/Atindama/EBSD-Restoration-Inpainting-and-Denoising/assets/121004801/2a007d04-c426-4853-bd22-afd3c4e0e9ca)
![ipf_damaged](https://github.com/Atindama/EBSD-Restoration-Inpainting-and-Denoising/assets/121004801/2a13a69b-7ca1-44c6-8938-81d1a400c3f1)
![ipf_hybrid](https://github.com/Atindama/EBSD-Restoration-Inpainting-and-Denoising/assets/121004801/6f03598d-0284-46f8-8f76-7397c4e28bd1)
![denoised_ipf_hybrid_oneblock](https://github.com/Atindama/EBSD-Restoration-Inpainting-and-Denoising/assets/121004801/e16a3c27-75aa-4f4a-a5f4-e3d400e0b6e6)

Clean       Noisy      Missing      Inpainted       Denoised

