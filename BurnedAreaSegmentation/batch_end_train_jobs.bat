@ECHO OFF

Rem This batch file executes kubectl commands to delete training jobs

::echo %kubectl%
SET exp_list=2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18

(for %%a in  (%exp_list%) do (
	ech0 %%a
	
	kubectl delete -f experiments_2\exp%%a/job_exp%%a_deeplabv3_img.yaml
	kubectl delete -f experiments_2\exp%%a/job_exp%%a_deeplabv3_tci.yaml
	kubectl delete -f experiments_2\exp%%a/job_exp%%a_deeplabv3plus_img.yaml
	kubectl delete -f experiments_2\exp%%a/job_exp%%a_deeplabv3plus_tci.yaml

	kubectl delete -f experiments\exp%%a/job_exp%%a_unet_img.yaml
	kubectl delete -f experiments\exp%%a/job_exp%%a_unet_tci.yaml
	kubectl delete -f experiments\exp%%a/job_exp%%a_unetplus_img.yaml
	kubectl delete -f experiments\exp%%a/job_exp%%a_unetplus_tci.yaml
))

echo "batch complete"