@ECHO OFF

Rem This batch file executes kubectl commands to create training jobs

::echo %kubectl%
SET exp_list=2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18

(for %%a in  (%exp_list%) do (
	echo %%a
	
	kubectl create -f experiments\exp%%a/job_exp%%a_deeplab_img.yaml
	kubectl create -f experiments\exp%%a/job_exp%%a_deeplab_tci.yaml
	kubectl create -f experiments\exp%%a/job_exp%%a_deeplab_img_pretrained.yaml
	kubectl create -f experiments\exp%%a/job_exp%%a_deeplab_tci_pretrained.yaml

	kubectl create -f experiments\exp%%a/job_exp%%a_unet_img.yaml
	kubectl create -f experiments\exp%%a/job_exp%%a_unet_tci.yaml
	kubectl create -f experiments\exp%%a/job_exp%%a_unet_img_pretrained.yaml
	kubectl create -f experiments\exp%%a/job_exp%%a_unet_tci_pretrained.yaml
))

echo "batch complete"