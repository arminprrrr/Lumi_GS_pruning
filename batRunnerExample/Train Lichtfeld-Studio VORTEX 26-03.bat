setlocal

set "LUMI_GS_PRUNING_FILE=D:\Vortex\pluginJsonFile.json"


D:\LichtFeld-Studio_Windows_v0.5.0\bin\LichtFeld-Studio.exe -d D:\Vortex\Muschelgriff -o D:\Vortex\Muschelgriff\LFS-mcmc12 --python-script="D:\Vortex\run_lumi_gs_pruning.py" --config=C:\Users\armin\Downloads\LfsConfigVortex_26-03.json --headless

@pause
